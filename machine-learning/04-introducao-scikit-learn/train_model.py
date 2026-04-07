"""
train_model.py

🔥 PIPELINE COMPLETO DE MACHINE LEARNING (VERSÃO "VIVA")

Se você entender esse arquivo profundamente, você entende:
→ como um modelo nasce
→ como ele é avaliado
→ como ele é escolhido
→ como ele é preparado para produção

Esse script simula um cenário REAL de engenharia de ML.
"""

# =========================
# IMPORTS
# =========================
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# =========================
# 1. GERAÇÃO DOS DADOS
# =========================
"""
Aqui estamos simulando um dataset.

💡 VIDA REAL:
Você NÃO cria dados assim — eles vêm de:
- banco de dados
- APIs
- data lake
- sensores

Mas isso aqui é PERFEITO pra aprender.

🔥 IMPORTANTE:
Você está definindo a "realidade" do problema manualmente.
Ou seja: você SABE qual é a fórmula verdadeira.
"""

np.random.seed(42)  # garante reprodutibilidade (ESSENCIAL em ML)
n_samples = 120     # número de casas

# Features (variáveis independentes)
size = np.random.uniform(50, 200, n_samples)      # tamanho da casa
rooms = np.random.randint(1, 5, n_samples)        # número de quartos
age = np.random.uniform(0, 30, n_samples)         # idade do imóvel
distance = np.random.uniform(1, 20, n_samples)    # distância do centro
garage = np.random.randint(0, 2, n_samples)       # tem garagem? (0 ou 1)

# Monta matriz de features (X)
X = np.column_stack((size, rooms, age, distance, garage))

print('-------------------------------')
print('X (features): ', X)
print('-------------------------------')

"""
🎯 Agora criamos o TARGET (y)

Essa é a parte mais importante:
→ estamos dizendo como o mundo funciona

Preço depende de:
+ tamanho (positivo)
+ quartos (positivo)
- idade (negativo)
- distância (negativo)
+ garagem (positivo)

🔥 E adicionamos RUÍDO (mundo real nunca é perfeito)
"""

y = (
    3000 * size +
    10000 * rooms -
    2000 * age -
    5000 * distance +
    15000 * garage +
    np.random.randn(n_samples) * 20000  # ruído
)

print('-------------------------------')
print('y (target): ', y)
print('-------------------------------')

feature_names = ["size", "rooms", "age", "distance", "garage"]


# =========================
# 2. SPLIT TREINO / TESTE
# =========================
"""
Aqui separamos o que o modelo vê vs o que ele NUNCA viu.

💡 IDEIA:
- treino → aprender
- teste → prova final

🔥 ERRO COMUM:
Treinar e testar no mesmo dado → overfitting ilusório
"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 3. PIPELINES
# =========================
"""
Pipeline = esteira de produção do modelo

💡 Por que isso é CRUCIAL?

Evita erros tipo:
- esquecer normalização
- aplicar transformação diferente em produção

🔥 REGRA DE OURO:
"Tudo que acontece no treino, TEM que acontecer na predição"
"""

models = {
    "Linear": Pipeline([
        ("scaler", StandardScaler()),   # normaliza dados (muito importante)
        ("model", LinearRegression())   # modelo linear puro
    ]),

    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))     # penaliza coeficientes grandes (evita overfitting)
    ]),

    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1))     # pode zerar coeficientes (feature selection automática)
    ]),

    """
    RandomForest é diferente:
    - não precisa de normalização
    - aprende relações NÃO LINEARES
    - muito usado em produção
    """
    "RandomForest": RandomForestRegressor(n_estimators=100)
}

results = {}


# =========================
# 4. TREINAMENTO E AVALIAÇÃO
# =========================
"""
Aqui acontece a mágica.

Para cada modelo:
→ treina
→ testa
→ valida com cross-validation

🔥 CROSS-VALIDATION:
Simula vários "mundos possíveis"
→ muito mais confiável que um único split
"""

for name, model in models.items():

    # Treinamento
    model.fit(X_train, y_train)

    # Predição
    y_pred = model.predict(X_test)

    # Métricas
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Validação cruzada (mais robusto)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    # Armazena tudo
    results[name] = {
        "model": model,
        "r2": r2,
        "rmse": rmse,
        "cv_mean": cv_scores.mean()
    }

    print(f"\n===== {name} =====")
    print("R²:", r2)
    print("RMSE:", rmse)
    print("CV R²:", cv_scores)
    print("CV Mean:", cv_scores.mean())


# =========================
# 5. ESCOLHA DO MELHOR MODELO
# =========================
"""
Aqui você toma uma decisão de negócio.

💡 Por que usar CV mean?
Porque é mais estável.

🔥 IMPORTANTE:
Melhor modelo ≠ modelo mais complexo
"""

best_model_name = max(results, key=lambda x: results[x]["cv_mean"])
best_model = results[best_model_name]["model"]

print("\n🏆 MELHOR MODELO:", best_model_name)


# =========================
# 6. DIAGNÓSTICO (SAÚDE DO MODELO)
# =========================
"""
Aqui você vira DETETIVE.

Você não só quer acertar — você quer entender:
→ quando erra
→ como erra
→ por quê erra
"""

y_pred = best_model.predict(X_test)
residuals = y_test - y_pred

# Resíduos vs predição (procura padrões)
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.title("Resíduos vs Predição")
plt.show()

# Distribuição dos erros
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=20)
plt.title("Distribuição dos Resíduos")
plt.show()

# QQ Plot (normalidade dos erros)
plt.figure(figsize=(6,6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot")
plt.show()


# =========================
# 7. OUTLIERS
# =========================
"""
Outliers = pontos "estranhos"

🔥 PERIGO:
- podem destruir seu modelo
- podem indicar erro nos dados

Aqui usamos z-score:
→ quantos desvios padrão o erro está da média
"""

z_scores = np.abs(stats.zscore(residuals))
outliers = np.where(z_scores > 3)

print("\nOutliers encontrados:", len(outliers[0]))


# =========================
# 8. FEATURE IMPORTANCE
# =========================
"""
Quer saber o que mais influencia o modelo?

💡 Isso é MUITO usado em:
- explicabilidade (XAI)
- auditoria
- confiança do negócio
"""

if hasattr(best_model, "named_steps"):
    model_step = best_model.named_steps["model"]

    if hasattr(model_step, "coef_"):
        importance = np.abs(model_step.coef_)

        print("\nFeature Importance:")
        for name, imp in zip(feature_names, importance):
            print(f"{name}: {imp}")


# =========================
# 9. SALVAR MODELO
# =========================
"""
Agora entramos em PRODUÇÃO.

💡 joblib:
- salva modelo treinado
- permite carregar depois na API

🔥 REGRA:
Nunca treine modelo dentro da API
→ sempre carregue pronto
"""

joblib.dump(best_model, "model.pkl")

print("\n✅ Modelo salvo como model.pkl")
