"""Ponto de entrada CLI do laboratorio Adult Income: Model Evaluation & Fairness V2."""

import argparse
import sys
from pathlib import Path

from src.config import CANONICAL_DATASET_PATH, REPORTS_DIR
from src.data.loader import exportar_dataset_canonico_de_openml
from src.pipeline import executar_pipeline


def _fmt_opcional(valor: float | None, casas: int = 5) -> str:
    """Formata numero decimal opcional sem falhar quando a metrica e indefinida."""
    if valor is None:
        return "Indefinido"
    return f"{float(valor):.{casas}f}"


def _rotulo_escopo_curto(escopo: str) -> str:
    """Resume o escopo de comparacao de EOD/EOdds para exibicao clara no terminal."""
    if escopo == "integral_todos_os_grupos":
        return "integral"
    if escopo.startswith("parcial"):
        return "PARCIAL"
    return "indefinido"


def construir_parser() -> argparse.ArgumentParser:
    """Constroi o parser de argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description=(
            "Adult Income: Model Evaluation & Fairness V2 — "
            "laboratorio reproduzivel de avaliacao de modelos e auditoria de Responsible AI."
        )
    )
    parser.add_argument(
        "--source",
        choices=["local", "openml"],
        default="local",
        help="Origem dos dados: 'local' (CSV canonico verificado por SHA-256) ou 'openml' (fetch_openml).",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Caminho opcional para arquivo CSV alternativo (usado com --source local).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Diretorio onde os 14 relatorios serao gravados (padrao: reports/).",
    )
    parser.add_argument(
        "--no-verify-hash",
        action="store_true",
        help="Desativa verificacao do SHA-256 oficial (necessario ao usar datasets alternativos).",
    )
    parser.add_argument(
        "--export-canonical",
        action="store_true",
        help="Regenera data/adult_openml_1590_v2.csv a partir do OpenML ID 1590 v2 e encerra.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Executa a CLI e retorna codigo de saida padronizado (0=sucesso, 1=falha, 2=entrada invalida)."""
    parser = construir_parser()
    args = parser.parse_args(argv)

    try:
        if args.export_canonical:
            destino = args.dataset_path or CANONICAL_DATASET_PATH
            info = exportar_dataset_canonico_de_openml(caminho_destino=destino)
            print("Copia local canonica exportada com sucesso:")
            print(f"  Arquivo: {info['caminho_relativo']}")
            print(f"  Linhas:  {info['linhas']}")
            print(f"  Bytes:   {info['tamanho_bytes']}")
            print(f"  SHA-256: {info['sha256']}")
            return 0

        resultado = executar_pipeline(
            fonte=args.source,
            caminho_dataset=args.dataset_path,
            diretorio_saida=args.output_dir,
            verificar_hash=not args.no_verify_hash,
            exportar_relatorios=True,
        )

        metricas = resultado["metricas_teste"]
        cm = metricas["matriz_confusao"]
        fair_sex = resultado["auditoria_fairness"]["por_atributo"]["sex"]["resumo"]
        fair_race = resultado["auditoria_fairness"]["por_atributo"]["race"]["resumo"]
        rec = resultado["reconciliacao"]

        print("=== Adult Income: Model Evaluation & Fairness V2 ===")
        print(f"Fonte:               {resultado['metadados_carga']['caminho_relativo']}")
        print(f"Modelo selecionado:  {resultado['resumo_selecao']['modelo_selecionado']}")
        print(
            f"Teste (N={metricas['total_registros']}): "
            f"Acc={_fmt_opcional(metricas['accuracy'], 5)} | "
            f"Rec={_fmt_opcional(metricas['recall'], 5)} | "
            f"Prec={_fmt_opcional(metricas['precision'], 5)} | "
            f"F1={_fmt_opcional(metricas['f1'], 5)}"
        )
        print(
            f"Matriz de confusao:  TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}"
        )
        print(
            f"Fairness (sex):      EOD={_fmt_opcional(fair_sex['equal_opportunity_difference'], 6)} "
            f"[{_rotulo_escopo_curto(fair_sex['equal_opportunity_escopo'])}] | "
            f"EOdds={_fmt_opcional(fair_sex['equalized_odds_difference'], 6)} "
            f"[{_rotulo_escopo_curto(fair_sex['equalized_odds_escopo'])}]"
        )
        print(
            f"Fairness (race):     EOD={_fmt_opcional(fair_race['equal_opportunity_difference'], 6)} "
            f"[{_rotulo_escopo_curto(fair_race['equal_opportunity_escopo'])}] | "
            f"EOdds={_fmt_opcional(fair_race['equalized_odds_difference'], 6)} "
            f"[{_rotulo_escopo_curto(fair_race['equalized_odds_escopo'])}]"
        )
        print(
            f"Validacao matematica: {rec['total_aprovadas']}/{rec['total_checagens']} aprovadas "
            f"({rec['total_reprovadas']} reprovadas, {rec['total_nao_aplicaveis']} N/A)"
        )
        print(f"Relatorios gerados:  {len(resultado['arquivos_gerados'])} arquivo(s)")
        print(f"Duracao:             {resultado['duracao_segundos']:.2f}s")
        return 0
    except (FileNotFoundError, ValueError) as erro:
        print(f"Erro de entrada/validacao: {erro}", file=sys.stderr)
        return 2
    except Exception as erro:
        print(f"Erro durante a execucao do pipeline: {erro}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
