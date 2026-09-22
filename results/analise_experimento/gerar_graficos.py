"""
Script para gerar gráficos de barras dos resultados experimentais do TCC.
Padrão acadêmico: layout limpo, tipografia legível, cores sóbrias e consistentes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Configurar matplotlib para padrão acadêmico
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'grid.alpha': 0.3,
})

# Mapeamento de nomes das variantes
variantes_map = {
    'baseline': '(a)',
    'baseline_traduzido': '(b)',
    'topk_aumentado': '(c)',
    'topk_aumentado_traduzido': '(d)'
}

# Carregar dados - usar caminhos relativos ao diretório do script
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

df_comparativo = pd.read_csv(os.path.join(script_dir, 'comparativo_variantes.csv'))
df_retrieval = pd.read_csv(os.path.join(script_dir, 'resumo_metricas_retrieval.csv'))
df_generation = pd.read_csv(os.path.join(script_dir, 'resumo_metricas_generation.csv'))

# Diretório de saída
output_dir = os.path.join(script_dir, 'figures')
os.makedirs(output_dir, exist_ok=True)

# Definir cores sóbrias e consistentes
cores = {
    '(a)': '#2E4057',  # Azul escuro
    '(b)': '#5D6D7E',  # Cinza azulado
    '(c)': '#C0392B',  # Vermelho escuro
    '(d)': '#839192'   # Cinza claro
}

# ============================================================
# GRÁFICO 1: RETRIEVAL - Comparativo por Métrica
# ============================================================

metricas_retrieval = ['hit_rate_at_k', 'recall_at_k', 'precision_at_k', 'mrr', 'similaridade_media_chunks']
nomes_metricas_retrieval = [r'$\mathit{Hit\ Rate}@K$', r'$\mathit{Recall}@K$', r'$\mathit{Precision}@K$', r'$\mathit{MRR}$', 'Similaridade']

# Filtrar dados de retrieval
df_retrieval_resumo = df_retrieval.groupby(['variacao_retrieval', 'metric'])['mean'].mean().reset_index()

# Criar gráfico
fig, ax = plt.subplots(figsize=(10, 6))

# Configurar posição das barras
x = np.arange(len(metricas_retrieval))
largura = 0.2

# Plotar barras para cada variante
for i, (variacao, nome_variacao) in enumerate(variantes_map.items()):
    dados_variacao = df_retrieval_resumo[df_retrieval_resumo['variacao_retrieval'] == variacao]
    valores = []
    for metrica in metricas_retrieval:
        valor = dados_variacao[dados_variacao['metric'] == metrica]['mean'].values
        valores.append(valor[0] if len(valor) > 0 else 0)

    barras = ax.bar(x + i * largura, valores, largura,
                   label=nome_variacao,
                   color=cores[nome_variacao],
                   edgecolor='black',
                   linewidth=0.5,
                   alpha=0.8)

    # Adicionar valores nas barras
    for bar, val in zip(barras, valores):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)

# Configurações do gráfico
ax.set_xlabel('Métricas')
ax.set_ylabel(r'$\mathit{Score}$ Médio')
ax.set_title(r'Retrieval - Comparativo por Métrica', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x + largura * 1.5)
ax.set_xticklabels(nomes_metricas_retrieval, rotation=0)
ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
ax.yaxis.grid(True, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Ajustar limites do eixo Y
ax.set_ylim(0, 0.65)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'retrieval_metricas_comparativo.svg'), dpi=300, facecolor='white', edgecolor='none')
plt.savefig(os.path.join(output_dir, 'retrieval_metricas_comparativo.png'), dpi=300, facecolor='white', edgecolor='none')
plt.close()

# ============================================================
# GRÁFICO 2: GENERATION - Comparativo por Métrica
# ============================================================

metricas_generation = ['groundedness_score', 'corretude_score', 'completude_score',
                      'clareza_score', 'precisao_tecnica_score', 'alucinacao_score']
nomes_metricas_generation = [r'$\mathit{Groundedness}$', 'Corretude', 'Completude',
                            'Clareza', 'Precisão Técnica', 'Alucinação']

# Filtrar dados de generation
df_generation_resumo = df_generation.groupby(['variacao_retrieval', 'metric'])['mean'].mean().reset_index()

# Criar gráfico
fig, ax = plt.subplots(figsize=(12, 6))

# Configurar posição das barras
x = np.arange(len(metricas_generation))
largura = 0.2

# Plotar barras para cada variante
for i, (variacao, nome_variacao) in enumerate(variantes_map.items()):
    dados_variacao = df_generation_resumo[df_generation_resumo['variacao_retrieval'] == variacao]
    valores = []
    for metrica in metricas_generation:
        valor = dados_variacao[dados_variacao['metric'] == metrica]['mean'].values
        valores.append(valor[0] if len(valor) > 0 else 0)

    barras = ax.bar(x + i * largura, valores, largura,
                   label=nome_variacao,
                   color=cores[nome_variacao],
                   edgecolor='black',
                   linewidth=0.5,
                   alpha=0.8)

    # Adicionar valores nas barras
    for bar, val in zip(barras, valores):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)

# Configurações do gráfico
ax.set_xlabel('Métricas')
ax.set_ylabel(r'$\mathit{Score}$ Médio')
ax.set_title(r'Generation - Comparativo por Métrica', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x + largura * 1.5)
ax.set_xticklabels(nomes_metricas_generation, rotation=15)
ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
ax.yaxis.grid(True, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Ajustar limites do eixo Y
ax.set_ylim(0, 5.0)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'generation_metricas_comparativo.svg'), dpi=300, facecolor='white', edgecolor='none')
plt.savefig(os.path.join(output_dir, 'generation_metricas_comparativo.png'), dpi=300, facecolor='white', edgecolor='none')
plt.close()

# ============================================================
# GRÁFICO 3: RANKING COMPOSTO - Score Médio Geral
# ============================================================

# Calcular score médio por variante
scores_retrieval = df_retrieval_resumo.groupby('variacao_retrieval')['mean'].mean()
scores_generation = df_generation_resumo.groupby('variacao_retrieval')['mean'].mean()

# Ponderar: 50% retrieval + 50% generation
scores_compostos = {}
for variacao in variantes_map.keys():
    if variacao in scores_retrieval.index and variacao in scores_generation.index:
        scores_compostos[variacao] = (scores_retrieval[variacao] + scores_generation[variacao]) / 2

# Criar DataFrame para ranking
df_ranking = pd.DataFrame({
    'variacao': list(scores_compostos.keys()),
    'score_composto': list(scores_compostos.values())
})
df_ranking['nome_variacao'] = df_ranking['variacao'].map(variantes_map)
df_ranking['variacao'] = pd.Categorical(
    df_ranking['variacao'],
    categories=list(variantes_map.keys()),
    ordered=True,
)
df_ranking = df_ranking.sort_values('variacao')

# Criar gráfico
fig, ax = plt.subplots(figsize=(8, 6))

# Coordenadas x
x = range(len(df_ranking))
scores = df_ranking['score_composto'].values
nomes = df_ranking['nome_variacao'].values
cores_grafico = [cores[nome] for nome in nomes]

barras = ax.bar(x, scores, color=cores_grafico, edgecolor='black', linewidth=0.5, alpha=0.8)

# Adicionar valores e rótulos nas barras
for i, (bar, score, nome) in enumerate(zip(barras, scores, nomes)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
           f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2, -0.08,
           nome, ha='center', va='top', fontsize=11)

# Configurações do gráfico
ax.set_ylabel(r'$\mathit{Score}$ Composto (Média Bruta)', fontsize=11)
ax.set_title(r'Ranking Composto - Desempenho Geral por Variante', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks([])
ax.set_ylim(0, max(scores) * 1.15 if len(scores) else 1.0)
ax.yaxis.grid(True, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Adicionar linhas divisoras
for i in range(len(df_ranking) - 1):
    ax.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ranking_composto.svg'), dpi=300, facecolor='white', edgecolor='none')
plt.savefig(os.path.join(output_dir, 'ranking_composto.png'), dpi=300, facecolor='white', edgecolor='none')
plt.close()

# ============================================================
# GRÁFICO 4: RANKING COMPOSTO - Score Bruto x Normalizado
# ============================================================

# Normalização min–max apenas para este gráfico comparativo.
score_min = df_ranking['score_composto'].min()
score_max = df_ranking['score_composto'].max()
if score_max > score_min:
    df_ranking['score_normalizado'] = (
        (df_ranking['score_composto'] - score_min) / (score_max - score_min)
    )
else:
    df_ranking['score_normalizado'] = 0.0

fig, ax = plt.subplots(figsize=(9, 6))
posicoes = np.arange(len(df_ranking))
largura = 0.36
scores_brutos = df_ranking['score_composto'].values
scores_normalizados = df_ranking['score_normalizado'].values
cores_variantes = [cores[nome] for nome in df_ranking['nome_variacao'].values]
cores_variantes_clareadas = [
    tuple(0.55 * componente + 0.45 for componente in matplotlib.colors.to_rgb(cor))
    for cor in cores_variantes
]

barras_brutas = ax.bar(
    posicoes - largura / 2,
    scores_brutos,
    largura,
    label=r'$\mathit{Score}$ bruto',
    color=cores_variantes,
    edgecolor='black',
    linewidth=0.5,
    alpha=0.85,
)
barras_normalizadas = ax.bar(
    posicoes + largura / 2,
    scores_normalizados,
    largura,
    label=r'$\mathit{Score}$ normalizado',
    color=cores_variantes_clareadas,
    edgecolor='black',
    linewidth=0.5,
    alpha=0.85,
)

for bar, score in zip(barras_brutas, scores_brutos):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f'{score:.3f}',
        ha='center',
        va='bottom',
        fontsize=9,
    )

for bar, score in zip(barras_normalizadas, scores_normalizados):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f'{score:.3f}',
        ha='center',
        va='bottom',
        fontsize=9,
    )

ax.set_ylabel(r'$\mathit{Score}$ Composto', fontsize=11)
ax.set_title(
    r'Ranking Composto',
    fontsize=13,
    fontweight='bold',
    pad=15,
)
ax.set_xticks(posicoes)
ax.set_xticklabels(df_ranking['nome_variacao'].values)
ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
ax.set_ylim(0, max(scores_brutos) * 1.35 if len(scores_brutos) else 1.0)
ax.yaxis.grid(True, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ranking_composto_bruto_normalizado.svg'), dpi=300, facecolor='white', edgecolor='none')
plt.savefig(os.path.join(output_dir, 'ranking_composto_bruto_normalizado.png'), dpi=300, facecolor='white', edgecolor='none')
plt.close()

# ============================================================
# GRÁFICO 5: SCORE COMPOSTO GERAL POR MODELO
# ============================================================

df_consolidado = pd.read_csv(os.path.join(script_dir, 'resultados_consolidados.csv'))
nomes_modelos = {
    'deepseek.v3.2': r'$\mathit{DeepSeek\ V3}$',
    'us.meta.llama4-maverick-17b-instruct-v1:0': r'$\mathit{Llama\ 4\ Maverick}$',
    'mistral.mistral-large-3-675b-instruct': r'$\mathit{Mistral\ Large}$',
}
ordem_modelos = list(nomes_modelos.keys())

scores_modelos = (
    df_consolidado.groupby('modelo')['composite_quality_score']
    .mean()
    .reindex(ordem_modelos)
    .dropna()
)

fig, ax = plt.subplots(figsize=(9, 6))
posicoes_modelos = np.arange(len(scores_modelos))
cores_modelos = ['#2E4057', '#5D6D7E', '#C0392B'][:len(scores_modelos)]
barras_modelos = ax.bar(
    posicoes_modelos,
    scores_modelos.values,
    color=cores_modelos,
    edgecolor='black',
    linewidth=0.5,
    alpha=0.85,
)

for barra, score in zip(barras_modelos, scores_modelos.values):
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height() + 0.01,
        f'{score:.3f}',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
    )

ax.set_ylabel(r'$\mathit{Score}$ Composto', fontsize=11)
ax.set_title(
    r'$\mathit{\mathbf{Score\ Composto}}$ Geral por Modelo',
    fontsize=13,
    fontweight='bold',
    pad=15,
)
ax.set_xticks(posicoes_modelos)
ax.set_xticklabels([nomes_modelos[modelo] for modelo in scores_modelos.index])
ax.set_ylim(0, max(scores_modelos.values) * 1.2 if len(scores_modelos) else 1.0)
ax.yaxis.grid(True, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'score_composto_modelos.svg'), dpi=300, facecolor='white', edgecolor='none')
plt.savefig(os.path.join(output_dir, 'score_composto_modelos.png'), dpi=300, facecolor='white', edgecolor='none')
plt.close()

print("Gráficos gerados com sucesso!")
print("- figures/retrieval_metricas_comparativo.svg/png")
print("- figures/generation_metricas_comparativo.svg/png")
print("- figures/ranking_composto.svg/png")
print("- figures/ranking_composto_bruto_normalizado.svg/png")
print("- figures/score_composto_modelos.svg/png")
