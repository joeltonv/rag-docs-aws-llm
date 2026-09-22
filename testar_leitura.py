import sqlite3


def inspecionar_banco_dados(db_path="knowledge_base.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("=" * 60)
    print("📊 DIAGNÓSTICO DA BASE DE CONHECIMENTO RAG")
    print("=" * 60)

    # 1. Verificar Softwares cadastrados
    cursor.execute("SELECT id, nome_software, nome_arquivo, source_path, arquivo_hash, ingestao_em FROM softwares")
    softwares = cursor.fetchall()

    print(f"\n📂 Softwares Encontrados ({len(softwares)}):")
    for sw in softwares:
        print(f"  • ID {sw[0]}: {sw[1]} (Arquivo: {sw[2]})")
        print(f"    ↳ Source path: {sw[3] or '[não informado]'}")
        print(f"    ↳ Hash do arquivo: {sw[4] or '[não informado]'}")
        print(f"    ↳ Ingestão em: {sw[5] or '[não informado]'}")

    # 2. Verificar a quantidade total de tópicos
    cursor.execute("SELECT COUNT(*) FROM topicos")
    total_topicos = cursor.fetchone()[0]

    print(f"\n📈 Estatísticas Gerais:")
    print(f"  • Total de Chunks/Tópicos criados: {total_topicos}")

    # 3. Listar uma amostra dos tópicos estruturados
    print("\n🔍 Amostra da Estrutura dos Primeiros 5 Tópicos:")
    print("-" * 60)

    cursor.execute("""
        SELECT t.hierarquia, t.titulo, LENGTH(t.conteudo_texto), t.conteudo_hash, t.ingestao_em
        FROM topicos t
        LIMIT 5
    """)
    topicos_amostra = cursor.fetchall()

    if not topicos_amostra:
        print("  [AVISO] Nenhum tópico estruturado foi encontrado no banco.")
    else:
        for topico in topicos_amostra:
            hierarquia, titulo, tam_texto, conteudo_hash, ingestao_em = topico
            print(f"📍 Tópico {hierarquia} - {titulo}")
            print(f"   ↳ Caracteres de Texto: {tam_texto}")
            print(f"   ↳ Hash do conteúdo: {conteudo_hash or '[não informado]'}")
            print(f"   ↳ Ingestão em: {ingestao_em or '[não informado]'}")
            print("-" * 60)

    conn.close()


def buscar_topicos_por_termo(db_path="knowledge_base.db", termo=None, limit=10):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("=" * 60)
    print(f"🔎 BUSCA NO BANCO DE CONHECIMENTO: '{termo}'")
    print("=" * 60)

    sql = """
        SELECT s.nome_software, t.hierarquia, t.titulo, t.conteudo_texto
        FROM topicos t
        JOIN softwares s ON t.software_id = s.id
        WHERE t.titulo LIKE ? OR t.conteudo_texto LIKE ?
        ORDER BY t.id
        LIMIT ?
    """
    cursor.execute(sql, (f"%{termo}%", f"%{termo}%", limit))
    resultados = cursor.fetchall()

    if not resultados:
        print(f"[NENHUM RESULTADO] Nenhum tópico encontrado para '{termo}'.")
    else:
        for idx, resultado in enumerate(resultados, start=1):
            nome_software, hierarquia, titulo, conteudo_texto = resultado
            print(f"\nResultado {idx} | Software: {nome_software} | {hierarquia}")
            print(f"Título: {titulo}")
            print(f"Conteúdo: {conteudo_texto[:800]}{'...' if len(conteudo_texto) > 800 else ''}")
            print("-" * 60)

    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Diagnóstico e consulta do banco de conhecimento RAG.'
    )
    parser.add_argument(
        '--db',
        default='knowledge_base.db',
        help='Caminho do banco SQLite a ser inspecionado.'
    )
    parser.add_argument(
        '--search',
        help='Termo de busca para consultar tópicos no banco.'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Número máximo de resultados de busca.'
    )

    args = parser.parse_args()

    if args.search:
        buscar_topicos_por_termo(args.db, args.search, args.limit)
    else:
        inspecionar_banco_dados(args.db)
