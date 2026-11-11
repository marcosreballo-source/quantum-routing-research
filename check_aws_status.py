"""Verifica se há recursos AWS/SageMaker ativos que possam gerar custos."""
import os
import boto3
from datetime import datetime, timezone
from pathlib import Path
import sys

REGION = "us-east-2"
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")


def _session():
    return boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=REGION,
    )


def main():
    session = _session()
    sm = session.client("sagemaker")

    # Training Jobs recentes
    jobs = sm.list_training_jobs(MaxResults=30, SortBy="CreationTime", SortOrder="Descending")
    print("=== Training Jobs recentes ===")
    for j in jobs["TrainingJobSummaries"]:
        print(f"  {j['TrainingJobName']}  Status: {j['TrainingJobStatus']}")

    # Detalhe do último job executado
    job_name = jobs["TrainingJobSummaries"][0]["TrainingJobName"]
    
    desc = get_job_status(sm, job_name)
    
    print_job_status(desc)
    
    # Endpoints ativos (geram custo!)
    endpoints = sm.list_endpoints(MaxResults=10)
    print(f"\n=== Endpoints ativos: {len(endpoints['Endpoints'])} ===")
    for ep in endpoints["Endpoints"]:
        print(f"  {ep['EndpointName']}  Status: {ep['EndpointStatus']}")
    if not endpoints["Endpoints"]:
        print("  (nenhum)")

    # Notebook Instances (geram custo se InService!)
    notebooks = sm.list_notebook_instances(MaxResults=10)
    print(f"\n=== Notebook Instances: {len(notebooks['NotebookInstances'])} ===")
    for nb in notebooks["NotebookInstances"]:
        print(f"  {nb['NotebookInstanceName']}  Status: {nb['NotebookInstanceStatus']}")
    if not notebooks["NotebookInstances"]:
        print("  (nenhuma)")

    # EC2 instances (verificar se alguma está rodando por conta do SageMaker)
    ec2 = session.client("ec2")
    instances = ec2.describe_instances(
        Filters=[{"Name": "instance-state-name", "Values": ["running", "pending"]}]
    )
    running = []
    for r in instances["Reservations"]:
        for i in r["Instances"]:
            running.append(f"  {i['InstanceId']}  Type: {i['InstanceType']}  State: {i['State']['Name']}")
    print(f"\n=== EC2 Instances rodando: {len(running)} ===")
    for r in running:
        print(r)
    if not running:
        print("  (nenhuma)")

    print("\n" + "=" * 50)
    all_clear = (
        all(j["TrainingJobStatus"] in ("Completed", "Failed", "Stopped") for j in jobs["TrainingJobSummaries"])
        and len(endpoints["Endpoints"]) == 0
        and all(nb["NotebookInstanceStatus"] == "Stopped" for nb in notebooks["NotebookInstances"])
        and len(running) == 0
    )
    if all_clear:
        print("OK: Nenhum recurso ativo gerando custos.")
    else:
        print("ATENÇÃO: Há recursos ativos! Verifique acima.")

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

if __name__ == "__main__":
    main()
