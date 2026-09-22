# run_once_download_models.py
import os
from sentence_transformers import SentenceTransformer

# Lista com os repositórios oficiais da Hugging Face
modelos_para_baixar = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-m3"
]

# Pasta de destino relativa ao diretório do script
script_dir = os.path.dirname(os.path.abspath(__file__))
pasta_destino_base = os.path.join(script_dir, "modelos")

print("📥 INICIANDO DOWNLOAD DE MODELOS (OFFLINE CACHE)")

# Cria a pasta raiz se não existir
os.makedirs(pasta_destino_base, exist_ok=True)

for nome_modelo in modelos_para_baixar:
    # Extrai apenas o nome final do modelo para criar a subpasta (ex: "bge-m3")
    nome_pasta = nome_modelo.split("/")[-1]
    caminho_local = os.path.join(pasta_destino_base, nome_pasta)

    print(f"⏳ Baixando [{nome_modelo}]...")
    # O SentenceTransformer fará o download automático caso não esteja no cache
    model = SentenceTransformer(nome_modelo)

    print(f"💾 Salvando [{nome_modelo}]...")
    model.save(caminho_local)

    print(f"✅ {nome_pasta} salvo com sucesso em: {caminho_local}")
    print("-" * 60)

print("\n🎉 Todos os downloads foram concluídos!")
