"""Consulta os gastos AWS do período de billing atual (mês corrente)."""
import os
import boto3
from datetime import datetime, timedelta

REGION = "us-east-1"  # Cost Explorer só funciona em us-east-1
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
    ce = session.client("ce")

    hoje = datetime.utcnow()
    inicio_mes = hoje.replace(day=1).strftime("%Y-%m-%d")
    fim = (hoje + timedelta(days=1)).strftime("%Y-%m-%d")

    # ── Total do mês ──
    total = ce.get_cost_and_usage(
        TimePeriod={"Start": inicio_mes, "End": fim},
        Granularity="MONTHLY",
        Metrics=["UnblendedCost", "UsageQuantity"],
    )

    print("=" * 55)
    print(f"  AWS — Gastos do período: {inicio_mes} a {fim}")
    print("=" * 55)

    for result in total["ResultsByTime"]:
        cost = result["Total"]["UnblendedCost"]
        print(f"\n  Total do mês:  ${float(cost['Amount']):.4f} {cost['Unit']}")

    # ── Por serviço ──
    por_servico = ce.get_cost_and_usage(
        TimePeriod={"Start": inicio_mes, "End": fim},
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )

    print("\n  ─── Detalhamento por serviço ───")
    servicos = []
    for result in por_servico["ResultsByTime"]:
        for group in result["Groups"]:
            nome = group["Keys"][0]
            valor = float(group["Metrics"]["UnblendedCost"]["Amount"])
            if valor > 0.0001:
                servicos.append((valor, nome))

    servicos.sort(reverse=True)
    for valor, nome in servicos:
        print(f"    ${valor:>10.4f}  {nome}")

    if not servicos:
        print("    (nenhum gasto registrado ainda)")

    # ── Por dia (últimos 7 dias ou desde o início do mês) ──
    inicio_diario = max(
        hoje.replace(day=1),
        hoje - timedelta(days=7),
    ).strftime("%Y-%m-%d")

    por_dia = ce.get_cost_and_usage(
        TimePeriod={"Start": inicio_diario, "End": fim},
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
    )

    print("\n  ─── Gastos diários recentes ───")
    for result in por_dia["ResultsByTime"]:
        dia = result["TimePeriod"]["Start"]
        valor = float(result["Total"]["UnblendedCost"]["Amount"])
        barra = "█" * int(valor * 50) if valor > 0 else ""
        print(f"    {dia}  ${valor:>8.4f}  {barra}")

    # ── Previsão para o mês ──
    try:
        forecast = ce.get_cost_forecast(
            TimePeriod={"Start": fim, "End": (hoje.replace(day=28) + timedelta(days=4)).replace(day=1).strftime("%Y-%m-%d")},
            Metric="UNBLENDED_COST",
            Granularity="MONTHLY",
        )
        prev = float(forecast["Total"]["Amount"])
        print(f"\n  Previsão para o mês: ${prev:.4f} {forecast['Total']['Unit']}")
    except Exception:
        # Previsão pode falhar se não há dados suficientes
        pass

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()
