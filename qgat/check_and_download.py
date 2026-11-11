"""
Verifica o status de um SageMaker Training Job e, se concluído,
baixa os artefatos (model.tar.gz) + logs completos do CloudWatch.

Funciona **offline**: lança o job com --no-wait, desliga o PC,
depois roda este script para buscar os resultados.

Uso:
    python check_and_download.py                 # lê o job de .last_job.txt
    python check_and_download.py QGAT-20260305-2100-100-100-2026-03-05-21-00-38-123
    python check_and_download.py --status-only    # só mostra status, não baixa
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import botocore
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# ======================== CONFIG ========================
REGION = "us-east-2"
OUTPUT_BASE = Path("./output")
LOCAL_MLRUNS_DIR = Path("./mlruns")
LAST_JOB_FILE = Path(".last_job.txt")
LOG_GROUP = "/aws/sagemaker/TrainingJobs"
# ========================================================


AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")


def _session():
    return boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=REGION,
    )
    
sm_client = _session().client("sagemaker")

def parse_args():
    p = argparse.ArgumentParser(
        description="Verifica status de um training job e baixa artefatos + logs"
    )
    p.add_argument(
        "job_name", nargs="?", default=None,
        help="Nome do training job (se omitido, lê de .last_job.txt)"
    )
    p.add_argument(
        "--status-only", action="store_true",
        help="Apenas mostra o status, não baixa nada"
    )
    p.add_argument(
        "--wait", action="store_true",
        help="Aguarda o job terminar (polling a cada 30s)"
    )
    p.add_argument(
        "--list", action="store_true", dest="list_jobs",
        help="Lista os últimos 10 training jobs"
    )
    return p.parse_args()


def resolve_job_name(args) -> str:
    """Resolve o nome do job a partir dos argumentos ou .last_job.txt."""
    if args.job_name:
        return args.job_name

    if LAST_JOB_FILE.exists():
        name = LAST_JOB_FILE.read_text(encoding="utf-8").strip()
        if name:
            print(f"Lendo job de {LAST_JOB_FILE}: {name}")
            return name

    print("ERRO: Nenhum job_name fornecido e .last_job.txt não encontrado.")
    print("  Uso: python check_and_download.py <nome-do-job>")
    print("  Ou lance com: python launch_sagemaker.py --no-wait")
    sys.exit(1)


def list_recent_jobs(sm_client) -> None:
    """Lista os últimos 10 training jobs."""
    resp = sm_client.list_training_jobs(
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=10,
    )

    jobs = resp.get("TrainingJobSummaries", [])
    if not jobs:
        print("Nenhum training job encontrado.")
        return

    print(f"\n{'Nome do Job':<65} {'Status':<12} {'Criado em'}")
    print("-" * 100)
    for j in jobs:
        name = j["TrainingJobName"]
        status = j["TrainingJobStatus"]
        created = j["CreationTime"].strftime("%Y-%m-%d %H:%M:%S")
        print(f"{name:<65} {status:<12} {created}")
    print()


def get_job_status(sm_client, job_name: str) -> dict:
    """Busca descrição completa do job."""
    try:
        return sm_client.describe_training_job(TrainingJobName=job_name)
    except sm_client.exceptions.ClientError as e:
        if "ValidationException" in str(e) or "ResourceNotFound" in str(e):
            print(f"ERRO: Job '{job_name}' não encontrado.")
            sys.exit(1)
        raise


def print_job_status(desc: dict) -> str:
    """Imprime status detalhado e retorna o status string."""
    job_name = desc["TrainingJobName"]
    status = desc["TrainingJobStatus"]
    secondary = desc.get("SecondaryStatus", "—")
    instance_type = desc.get("ResourceConfig", {}).get("InstanceType", "?")
    created = desc.get("CreationTime")
    started = desc.get("TrainingStartTime")
    ended = desc.get("TrainingEndTime")

    print()
    print("=" * 62)
    print("  Q-GAT VRP — Status do Training Job")
    print("=" * 62)
    print(f"  Job:          {job_name}")
    print(f"  Status:       {status}")
    print(f"  Sub-status:   {secondary}")
    print(f"  Instância:    {instance_type}")

    if created:
        print(f"  Criado:       {created.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    if started:
        print(f"  Início:       {started.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    if ended:
        print(f"  Fim:          {ended.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Calcular duração
    if started:
        end_time = ended if ended else datetime.now(timezone.utc)
        duration = end_time - started
        total_secs = int(duration.total_seconds())
        hours, remainder = divmod(total_secs, 3600)
        minutes, secs = divmod(remainder, 60)
        print(f"  Duração:      {hours}h {minutes}m {secs}s")

    # Custo estimado (se billable)
    billable_secs = desc.get("BillableTimeInSeconds")
    if billable_secs:
        # Preços aproximados por hora em us-east-2
        price_map = {
            "ml.g4dn.xlarge": 0.736,
            "ml.g5.xlarge": 1.408,
            "ml.g5.2xlarge": 1.515,
            "ml.p3.2xlarge": 3.825,
        }
        price_per_sec = price_map.get(instance_type, 0.736) / 3600
        cost = billable_secs * price_per_sec
        print(f"  Tempo faturável: {billable_secs}s")
        print(f"  Custo estimado:  ${cost:.2f}")

    if status == "Failed":
        reason = desc.get("FailureReason", "desconhecido")
        print(f"\n  FALHA: {reason}")

    print("=" * 62)
    return status


def wait_for_completion(sm_client, job_name: str) -> dict:
    """Polling até o job sair de InProgress."""
    print(f"\nAguardando job '{job_name}' terminar (Ctrl+C para cancelar polling)...")
    last_secondary = ""

    while True:
        desc = get_job_status(sm_client, job_name)
        status = desc["TrainingJobStatus"]
        secondary = desc.get("SecondaryStatus", "")

        if secondary != last_secondary:
            ts = time.strftime("%H:%M:%S")
            print(f"  [{ts}] {status} / {secondary}")
            last_secondary = secondary

        if status in ("Completed", "Failed", "Stopped"):
            return desc

        time.sleep(30)


def download_cloudwatch_logs(logs_client, job_name: str) -> Path | None:
    """Baixa TODOS os logs do CloudWatch e salva em output/<job_name>/training.log."""
    job_dir = OUTPUT_BASE / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    log_file = job_dir / "training.log"

    print(f"\nBaixando logs completos do CloudWatch...")
    print(f"  Log group:  {LOG_GROUP}")

    # Descobrir log streams do job (o nome exato tem sufixo variável)
    try:
        streams_resp = logs_client.describe_log_streams(
            logGroupName=LOG_GROUP,
            logStreamNamePrefix=f"{job_name}/",
        )
        streams = [s["logStreamName"] for s in streams_resp.get("logStreams", [])]
    except Exception as e:
        print(f"  Erro ao buscar log streams: {e}")
        return None

    if not streams:
        print(f"  Nenhum log stream encontrado para '{job_name}'.")
        return None

    print(f"  Log streams encontrados: {len(streams)}")
    for s in streams:
        print(f"    {s}")

    try:
        all_events = []

        for stream_name in streams:
            kwargs = {
                "logGroupName": LOG_GROUP,
                "logStreamName": stream_name,
                "startFromHead": True,
            }

            while True:
                response = logs_client.get_log_events(**kwargs)
                events = response.get("events", [])

                if not events:
                    break

                all_events.extend(events)
                next_token = response.get("nextForwardToken")

                # Se o token não muda, não há mais eventos
                if next_token == kwargs.get("nextToken"):
                    break

                kwargs["nextToken"] = next_token

        if not all_events:
            print("  Nenhum log encontrado no CloudWatch.")
            return None

        # Ordenar por timestamp
        all_events.sort(key=lambda e: e["timestamp"])

        # Escrever logs em arquivo
        with open(log_file, "w", encoding="utf-8") as f:
            for event in all_events:
                ts = datetime.fromtimestamp(
                    event["timestamp"] / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")
                msg = event["message"].rstrip()
                f.write(f"[{ts}] {msg}\n")

        print(f"  {len(all_events)} linhas de log salvas em: {log_file}")
        return log_file

    except logs_client.exceptions.ResourceNotFoundException:
        print(f"  Log group/stream não encontrado no CloudWatch.")
        return None
    except Exception as e:
        print(f"  Erro ao buscar logs: {e}")
        return None


def download_and_extract(sm_client, s3_client, s3_uri: str, job_name: str) -> Path | None:
    """Baixa model.tar.gz do S3 e organiza em output/<job_name>/."""
    print(f"\nBaixando artefatos de: {s3_uri}")

    job_dir = OUTPUT_BASE / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    local_tar = job_dir / "model.tar.gz"

    s3_path = s3_uri.replace("s3://", "")
    bucket, key = s3_path.split("/", 1)

    # Gerar presigned URL e usar requests com retry automático
    s3_client_local = boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    url = s3_client_local.generate_presigned_url(
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
        # Windows pode travar em .trash — forçar permissão
        def _on_rm_error(func, path, exc_info):
            import stat
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(tmp_extract, onerror=_on_rm_error)
    tmp_extract.mkdir(parents=True)

    print(f"  Extraindo e organizando em: {job_dir}")
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=str(tmp_extract))

    # Organizar: checkpoints + config na raiz do job_dir
    checkpoints_dir = job_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    for pt_file in tmp_extract.rglob("*.pt"):
        shutil.copy2(str(pt_file), str(checkpoints_dir / pt_file.name))

    for cfg_file in tmp_extract.rglob("config.resolved.yaml"):
        shutil.copy2(str(cfg_file), str(job_dir / "config.resolved.yaml"))

    # Listar conteúdo organizado
    all_files = sorted(f for f in job_dir.rglob("*") if f.is_file() and "_tmp_extract" not in str(f))
    print(f"  {len(all_files)} arquivos organizados")
    for p in all_files:
        rel = p.relative_to(job_dir)
        print(f"    {rel}  ({p.stat().st_size:,} bytes)")

    return job_dir


def import_mlruns(job_dir: Path) -> bool:
    """Copia mlruns dos artefatos para o diretório local."""
    remote_mlruns = job_dir / "_tmp_extract" / "mlruns"
    if not remote_mlruns.exists():
        print("\n  Pasta 'mlruns' não encontrada nos artefatos.")
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


def print_summary(
    job_name: str,
    status: str,
    extract_dir: Path | None,
    mlflow_ok: bool,
    log_file: Path | None,
):
    print()
    print("=" * 62)
    print("  CONCLUÍDO — Resultados baixados com sucesso")
    print("=" * 62)

    if extract_dir:
        checkpoints = list((extract_dir / "checkpoints").glob("*.pt")) if (extract_dir / "checkpoints").exists() else []
        if checkpoints:
            print(f"\n  Checkpoints ({len(checkpoints)}):")
            for ckpt in sorted(checkpoints):
                print(f"    checkpoints/{ckpt.name}")

        config_file = extract_dir / "config.resolved.yaml"
        if config_file.exists():
            print(f"\n  Config resolvida: config.resolved.yaml")

    if log_file:
        print(f"\n  Log completo: {log_file}")

    if mlflow_ok:
        print(f"\n  MLflow runs: {LOCAL_MLRUNS_DIR.resolve()}")
        print()
        print("  Para visualizar gráficos e métricas:")
        print(f"    cd {LOCAL_MLRUNS_DIR.parent.resolve()}")
        print("    mlflow ui --port 5000")
        print("    Abra no navegador: http://localhost:5000")
    else:
        if extract_dir:
            print(f"\n  Artefatos em: {extract_dir}")

    print()


def main():
    args = parse_args()

    # Clientes AWS
    sm = _session().client("sagemaker")
    logs = _session().client("logs")
    s3 = _session().client("s3", region_name=REGION)

    # --list: mostrar jobs recentes
    if args.list_jobs:
        list_recent_jobs(sm)
        return

    # Resolver nome do job
    job_name = resolve_job_name(args)

    # Buscar status
    desc = get_job_status(sm, job_name)
    status = print_job_status(desc)

    # --status-only: parar aqui
    if args.status_only:
        return

    # --wait: polling até terminar
    if status == "InProgress":
        if args.wait:
            desc = wait_for_completion(sm, job_name)
            status = print_job_status(desc)
        else:
            print("\n  O job ainda está rodando.")
            print("  Use --wait para aguardar, ou volte depois.")
            print(f"    python check_and_download.py {job_name} --wait")
            return

    if status == "Failed":
        print("\n  Baixando logs para diagnóstico...")
        download_cloudwatch_logs(logs, job_name)
        sys.exit(1)

    if status == "Stopped":
        print("\n  Job foi parado manualmente.")
        download_cloudwatch_logs(logs, job_name)
        return

    # Status == "Completed" → baixar tudo
    if status == "Completed":
        # 1. Logs do CloudWatch
        log_file = download_cloudwatch_logs(logs, job_name)

        # 2. Artefatos do S3
        s3_uri = desc.get("ModelArtifacts", {}).get("S3ModelArtifacts")
        extract_dir = None
        if s3_uri:
            extract_dir = download_and_extract(sm, s3, s3_uri, job_name)
        else:
            print("\n  Nenhum artefato S3 encontrado.")

        # 3. Importar mlruns
        mlflow_ok = False
        if extract_dir:
            mlflow_ok = import_mlruns(extract_dir)

        # 4. Resumo
        print_summary(job_name, status, extract_dir, mlflow_ok, log_file)


if __name__ == "__main__":
    main()
