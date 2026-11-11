"""
Lança um SageMaker Training Job, AGUARDA até terminar,
baixa model.tar.gz do S3 e importa os mlruns no MLflow local.

Pré-requisitos (instalar no seu PC):
    pip install sagemaker boto3 mlflow

Uso:
    python launch_sagemaker.py
    python launch_sagemaker.py --config configs/light_test.yaml
    python launch_sagemaker.py --config configs/baseline_1k_val.yaml --data_size 20480 --val_size 4096
    python launch_sagemaker.py --instance ml.g5.xlarge --config configs/baseline_10k.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import shutil
from pathlib import Path

import yaml
import boto3
import botocore
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from sagemaker.pytorch.estimator import PyTorch
import sagemaker

from check_and_download import AWS_SECRET_ACCESS_KEY, AWS_ACCESS_KEY_ID


# ====================== VALORES PADRÃO ======================

REGION = "us-east-2"

# IAM Role com permissão SageMaker
DEFAULT_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "")

DEFAULT_INSTANCE_TYPE =  "ml.g4dn.2xlarge"   # 1x T4, 4 vCPU, 16 GB — ~$0.74/h em us-east-2
#"ml.g5.2xlarge"   1x A10G, 8 vCPU, 32 GB — ~$1.06/h em us-east-2 (requer limite aprovado)



#Número máximo de horas para o job, k * 3600 segundos. O job será interrompido se ultrapassar esse tempo.
MAX_RUN_SECONDS = 120 * 3600                 

# Framework no container SageMaker
PYTORCH_VERSION = "2.1"
PYTHON_VERSION = "py310"

# Diretórios locais para salvar saída
OUTPUT_BASE = Path("./output")
LOCAL_MLRUNS_DIR = Path("./mlruns")

# =============================================================


def parse_args():
    p = argparse.ArgumentParser(description="Lança treino no SageMaker e baixa resultados")
    p.add_argument("--config", default="configs/base.yaml",
                   help="Arquivo YAML de config do experimento")
    p.add_argument("--base-config", default="configs/base.yaml",
                   help="Arquivo YAML base")
    p.add_argument("--data_size", type=int, default=None,
                   help="Sobrescrever tamanho do dataset de treino")
    p.add_argument("--val_size", type=int, default=None,
                   help="Sobrescrever tamanho do dataset de validação")
    p.add_argument("--instance", default=DEFAULT_INSTANCE_TYPE,
                   help=f"Tipo de instância SageMaker (default: {DEFAULT_INSTANCE_TYPE})")
    p.add_argument("--role", default=DEFAULT_ROLE_ARN,
                   help="ARN do IAM Role para SageMaker")
    p.add_argument("--max-hours", type=float, default=MAX_RUN_SECONDS / 3600,
                   help="Tempo máximo do job em horas")
    p.add_argument("--no-download", action="store_true",
                   help="Não baixar artefatos ao final (só lançar)")
    p.add_argument("--no-wait", action="store_true",
                    help="Lança o job e sai imediatamente (modo offline)")
    p.add_argument("--project", default=None,
                   help="Nome do projeto (default: lido do YAML sagemaker.project ou 'bqc')")
    p.add_argument("--squad", default=None,
                   help="Nome do squad (default: lido do YAML sagemaker.squad ou 'qgat')")
    return p.parse_args()


def _resolve_dataset_sizes(args) -> tuple[int, int]:
    """Lê data_size e val_size dos configs YAML, aplicando overrides da CLI."""
    with open(args.base_config) as f:
        base = yaml.safe_load(f) or {}
    with open(args.config) as f:
        exp = yaml.safe_load(f) or {}

    ds = base.get("dataset", {})
    ds.update(exp.get("dataset", {}))

    data_size = args.data_size if args.data_size is not None else ds.get("data_size", 0)
    val_size = args.val_size if args.val_size is not None else ds.get("val_size", 0)
    return int(data_size), int(val_size)


def _resolve_sagemaker_names(args) -> tuple[str, str]:
    """Resolve project/squad from CLI args or YAML config."""
    with open(args.base_config) as f:
        base = yaml.safe_load(f) or {}
    with open(args.config) as f:
        exp = yaml.safe_load(f) or {}

    sm = base.get("sagemaker", {})
    sm.update(exp.get("sagemaker", {}))

    project = args.project if args.project is not None else sm.get("project", "NO-PROJECT")
    squad = args.squad if args.squad is not None else sm.get("squad", "NO-SQUAD")
    return project, squad


# ---- Mapeamento de abreviações para o tipo de crítico ----
# (Mantém critic types abreviados porque são longos e só aparecem quando baseline==critic)
_CRITIC_ABBREV: dict[str, str] = {
    "shared_conv1d":      "sc",
    "shared_fc":          "sf",
    "own_encoder_conv1d": "oec",
    "own_encoder_mlp":    "oem",
    # nomes legados (mapeados em critics.py)
    "lei_conv1d":         "sc",
    "kool_encoder":       "oem",
}

# Abreviações de topologia quantum para manter o nome curto
# (brickwall→bw, chain→chn, lambda→lmb; ring permanece "ring")
_TOPO_ABBREV: dict[str, str] = {
    "brickwall": "bw",
    "chain":     "chn",
    "lambda":    "lmb",
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge: valores de `override` sobrepõem os de `base` recursivamente."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_merged_config(args) -> dict:
    """Carrega base.yaml + config do experimento e devolve o dict mesclado."""
    with open(args.base_config) as f:
        base = yaml.safe_load(f) or {}
    with open(args.config) as f:
        exp = yaml.safe_load(f) or {}
    return _deep_merge(base, exp)


def _extract_specs_from_config(cfg: dict) -> dict:
    """Extrai specs relevantes para o nome do job a partir do config mesclado.
    
    Também resolve auto-baseline (null):
      - PPO → critic
      - Rollout → rollout
    """
    training = cfg.get("training") or {}
    quantum  = cfg.get("quantum")  or {}

    method   = str(training.get("method")   or "ppo").lower()
    baseline = training.get("baseline")
    
    # Auto-resolve: null baseline depends on method
    if not baseline:
        if method == "ppo":
            baseline = "critic"
        elif method == "rollout":
            baseline = "rollout"
        else:
            baseline = "none"
    
    baseline = str(baseline).lower()

    # Tipo do crítico — relevante quando baseline == "critic" OU quando há quantum.critic.enabled
    critic_abbrev = None
    critic_cfg = quantum.get("critic") or {}
    
    # Mostrar tipo de crítico se:
    #   1. baseline == "critic" (crítico como baseline), OU
    #   2. quantum.critic.enabled=true (crítico é parte da arquitetura mesmo que não seja baseline)
    if baseline == "critic" or critic_cfg.get("enabled", False):
        raw_type = critic_cfg.get("type") or "shared_conv1d"
        critic_abbrev = _CRITIC_ABBREV.get(raw_type, raw_type[:4])

    q_enabled  = bool(quantum.get("enabled", False))
    q_topology = str(quantum.get("topology") or "ring").lower()
    q_ansatz   = str(quantum.get("rotation") or "RYRZ").upper()

    return {
        "method":        method,
        "baseline":      baseline,
        "critic_abbrev": critic_abbrev,
        "q_enabled":     q_enabled,
        "q_topology":    q_topology,
        "q_ansatz":      q_ansatz,
    }


def _build_sagemaker_tags(
    project: str,
    squad: str,
    additional_tags: list | None = None,
) -> list[dict]:
    """Constrói lista de tags AWS para rastreamento de custos.

    Tags padrão: `project` e `squad`.  Quaisquer `additional_tags` são
    mescladas sem duplicar chaves existentes (retrocompatível).
    """
    tags: list[dict] = [
        {"Key": "Project", "Value": project},
        {"Key": "Squad",   "Value": squad},
    ]
    if additional_tags:
        existing_keys = {t["Key"] for t in tags}
        for tag in additional_tags:
            if tag.get("Key") not in existing_keys:
                tags.append(tag)
                existing_keys.add(tag["Key"])
    return tags


def _build_job_name(cfg: dict, project: str, squad: str) -> str:
    """Gera nome do job a partir das specs do config.

    Formato:  {project}-{squad}-{method}-{baseline}-[{critic}]-{Q_flag}
    (Sem data — AWS SageMaker adiciona timestamp no sufixo automaticamente)

    Exemplos:
      bqc-qgat-ppo-none-C
      bqc-qgat-ppo-critic-sc-C
      bqc-qgat-rollout-rollout-C
      bqc-qgat-rollout-critic-oem-Qring-RYRZ
      bqc-qgat-ppo-critic-oec-Qbw-RXRYRZ
    """
    specs = _extract_specs_from_config(cfg)
    parts: list[str] = [project, squad, specs["method"], specs["baseline"]]

    if specs["critic_abbrev"] is not None:
        parts.append(specs["critic_abbrev"])

    if specs["q_enabled"]:
        topo = _TOPO_ABBREV.get(specs["q_topology"], specs["q_topology"])
        parts.append(f"Q{topo}-{specs['q_ansatz']}")
    else:
        parts.append("C")

    name = "-".join(parts)

    # SageMaker: base_job_name ≤ 39 chars (AWS acrescenta sufixo ~24 chars → total ≤ 63)
    if len(name) > 39:
        name = name[:36] + "---"
        print(f"  Aviso: nome base do job truncado para 39 chars: {name!r}")

    return name


def launch_and_wait(args) -> tuple[PyTorch, sagemaker.Session]:
    """Lança o Training Job e bloqueia até finalizar."""

    os.environ["AWS_DEFAULT_REGION"] = REGION
    boto_session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                region_name=REGION)
    
    
    session = sagemaker.Session(boto_session=boto_session)
    bucket = session.default_bucket()

    # ---------- Hiperparâmetros ----------
    hyperparameters = {
        "CONFIG_FILE": str(args.config).replace("\\", "/"),
        "BASE_CONFIG": str(args.base_config).replace("\\", "/"),
    }
    if args.data_size is not None:
        hyperparameters["DATA_SIZE"] = str(args.data_size)
    if args.val_size is not None:
        hyperparameters["VAL_SIZE"] = str(args.val_size)

    # ---------- Nome do Job + Tags ----------
    cfg = _load_merged_config(args)
    data_size, val_size = _resolve_dataset_sizes(args)
    sm_cfg  = cfg.get("sagemaker") or {}
    project = args.project if args.project is not None else sm_cfg.get("project", "NO-PROJECT")
    squad   = args.squad   if args.squad   is not None else sm_cfg.get("squad",   "NO-SQUAD")
    job_name = _build_job_name(cfg, project, squad)
    tags     = _build_sagemaker_tags(project, squad)

    print("=" * 62)
    print("  Q-GAT VRP — SageMaker Training Job")
    print("=" * 62)
    print(f"  Job name:     {job_name}")
    print(f"  Region:       {REGION}")
    print(f"  Bucket S3:    {bucket}")
    print(f"  AWSInstance:    {args.instance}")
    print(f"  Config:       {args.config}")
    print(f"  Data size:    {data_size}")
    print(f"  Val size:     {val_size}")
    print(f"  Max run:      {args.max_hours:.1f}h")
    print(f"  Role:         {args.role}")
    tag_str = ", ".join(f"{t['Key']}={t['Value']}" for t in tags)
    print(f"  Tags:         {tag_str}")
    print("=" * 62)
    print()

    if "<ACCOUNT_ID>" in args.role:
        print("ERRO: Edite o ROLE_ARN no topo do script (ou passe --role).")
        print("  Exemplo: arn:aws:iam::123456789012:role/SageMakerRole")
        sys.exit(1)

    estimator = PyTorch(
        entry_point="sagemaker_entry.py",
        source_dir=".",                       # envia todo o baseline_aws/
        base_job_name=job_name,               # {project}-{squad}-{method}-{baseline}-{Q}-{DDMMAA}
        role=args.role,
        instance_type=args.instance,
        instance_count=1,
        framework_version=PYTORCH_VERSION,
        py_version=PYTHON_VERSION,
        max_run=int(args.max_hours * 3600),
        sagemaker_session=session,
        hyperparameters=hyperparameters,
        tags=tags,                            # rastreamento de custos: project + squad
        # Nota: metric_definitions removido (incompatibilidade sagemaker)
        # As métricas são registradas no MLflow local dentro do container
    )

    if args.no_wait:
        print("Lançando Training Job (modo OFFLINE — not aguarda)...\n")
        estimator.fit(wait=False, logs="None")
    else:
        print("Lançando Training Job (wait=True, logs em tempo real)...\n")
        # wait=True  → bloqueia até o job terminar
        # logs="All" → transmite logs do CloudWatch para seu terminal
        estimator.fit(wait=True, logs="All")

    job_name = estimator.latest_training_job.name

    # Salvar nome do job para recuperação posterior
    last_job_file = Path(".last_job.txt")
    last_job_file.write_text(job_name, encoding="utf-8")
    print(f"  Nome do job salvo em: {last_job_file.resolve()}")

    if args.no_wait:
        print(f"\n  Job '{job_name}' launched com sucesso!")
        print(f"  O treinamento continua rodando no AWS mesmo com o PC desligado.")
        print(f"\n  Para verificar o status depois:")
        print(f"    python check_and_download.py")
        print(f"    python check_and_download.py {job_name}")
        return estimator, session

    status = estimator.latest_training_job.describe()["TrainingJobStatus"]
    print(f"\nJob '{job_name}' finalizado com status: {status}")
    return estimator, session


def download_and_extract(estimator) -> tuple[Path | None, str]:
    """Baixa model.tar.gz do S3 e organiza em output/<job_name>/."""

    model_data = estimator.model_data
    job_name = estimator.latest_training_job.name

    if not model_data:
        print("\nNenhum artefato encontrado no S3.")
        return None, job_name

    print(f"\nBaixando artefatos de: {model_data}")

    job_dir = OUTPUT_BASE / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    local_tar = job_dir / "model.tar.gz"

    # Parse S3 URI → bucket + key
    s3_path = model_data.replace("s3://", "")
    bucket, key = s3_path.split("/", 1)

    s3 = boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    # Gerar presigned URL e usar requests com retry automático
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=3600,
    )
    
    # Usar requests com retry automático para contornar 307 redirects
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=0.5, status_forcelist=[307, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    print(f"  Baixando com retry automático...")
    response = session.get(url, stream=True)
    response.raise_for_status()
    
    with open(str(local_tar), "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    size_mb = local_tar.stat().st_size / 1e6
    print(f"  Download completo: {local_tar} ({size_mb:.1f} MB)")

    # Extrair em diretório temporário
    tmp_extract = job_dir / "_tmp_extract"
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)
    tmp_extract.mkdir(parents=True)

    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=str(tmp_extract))

    # Organizar artefatos: checkpoints + config na raiz do job_dir
    checkpoints_dir = job_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    for pt_file in tmp_extract.rglob("*.pt"):
        shutil.copy2(str(pt_file), str(checkpoints_dir / pt_file.name))

    for cfg_file in tmp_extract.rglob("config.resolved.yaml"):
        shutil.copy2(str(cfg_file), str(job_dir / "config.resolved.yaml"))

    # Manter mlruns extraído para importação
    extracted_mlruns = tmp_extract / "mlruns"

    # Listar conteúdo organizado
    all_files = sorted(f for f in job_dir.rglob("*") if f.is_file() and "_tmp_extract" not in str(f))
    print(f"  {len(all_files)} arquivos organizados em: {job_dir}")
    for p in all_files:
        rel = p.relative_to(job_dir)
        print(f"    {rel}  ({p.stat().st_size:,} bytes)")

    return job_dir, job_name


def import_mlruns(job_dir: Path) -> bool:
    """Copia a pasta mlruns dos artefatos para o diretório local."""

    remote_mlruns = job_dir / "_tmp_extract" / "mlruns"
    if not remote_mlruns.exists():
        print("\nPasta 'mlruns' não encontrada nos artefatos.")
        print("  O MLflow pode não ter logado durante o treinamento.")
        return False

    print(f"\nImportando mlruns para: {LOCAL_MLRUNS_DIR.resolve()}")

    if LOCAL_MLRUNS_DIR.exists():
        for item in remote_mlruns.iterdir():
            dst = LOCAL_MLRUNS_DIR / item.name
            if item.is_dir():
                shutil.copytree(str(item), str(dst), dirs_exist_ok=True)
            else:
                shutil.copy2(str(item), str(dst))
    else:
        shutil.copytree(str(remote_mlruns), str(LOCAL_MLRUNS_DIR))

    experiments = [d for d in LOCAL_MLRUNS_DIR.iterdir() if d.is_dir() and d.name != ".trash"]
    print(f"  {len(experiments)} experiment(s) no MLflow local")

    # Limpar extração temporária
    tmp_extract = job_dir / "_tmp_extract"
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract, ignore_errors=True)

    return True


def print_summary(job_dir: Path | None, mlflow_ok: bool):
    print()
    print("=" * 62)
    print("  CONCLUÍDO")
    print("=" * 62)

    if job_dir:
        print(f"\n  Artefatos em: {job_dir}")

        checkpoints = list((job_dir / "checkpoints").glob("*.pt")) if (job_dir / "checkpoints").exists() else []
        if checkpoints:
            print(f"\n  Checkpoints ({len(checkpoints)}):")
            for ckpt in sorted(checkpoints):
                print(f"    checkpoints/{ckpt.name}")

        config_file = job_dir / "config.resolved.yaml"
        if config_file.exists():
            print(f"\n  Config resolvida: config.resolved.yaml")

    if mlflow_ok:
        print(f"\n  MLflow runs: {LOCAL_MLRUNS_DIR.resolve()}")
        print()
        print("  Para visualizar gráficos e métricas:")
        print(f"    cd {LOCAL_MLRUNS_DIR.parent.resolve()}")
        print( "    mlflow ui --port 5000")
        print( "    Abra no navegador: http://localhost:5000")
    else:
        if job_dir:
            print(f"\n  Artefatos em: {job_dir}")

    print()


def main():
    args = parse_args()

    # 0. Validar configs localmente antes de despachar para SageMaker
    from validate_config import validate_configs
    
    
    ok, errors = validate_configs(
        base_path=Path(args.base_config),
        config_path=Path(args.config),
        data_size=args.data_size,
        val_size=args.val_size,
    )
    if ok != True:
        print("\n" + "=" * 62)
        print("  ERROR: INVALID CONFIGURATION — job NOT dispatched")
        print("=" * 62)
        for err in errors:
            print(f"  • {err}")
        print()
        sys.exit(1)

    # 1. Lançar (e esperar, se não --no-wait)
    estimator, session = launch_and_wait(args)

    if args.no_wait:
        return

    if args.no_download:
        print("\n--no-download: pulando download de artefatos.")
        print(f"  Artefatos no S3: {estimator.model_data}")
        return

    # 2. Baixar e extrair
    job_dir, _ = download_and_extract(estimator)

    # 3. Importar mlruns localmente
    mlflow_ok = False
    if job_dir:
        mlflow_ok = import_mlruns(job_dir)

    # 4. Resumo final
    print_summary(job_dir, mlflow_ok)


if __name__ == "__main__":
    main()
