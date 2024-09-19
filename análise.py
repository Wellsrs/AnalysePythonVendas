import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Carregar o arquivo CSV para inspeção
file_path = r"C:\Users\hanie\Downloads\sales.csv"
df = pd.read_csv(file_path)

print(df)
# Exibir as primeiras linhas para entender os dados
df.head(), df.columns, df.info()

# --------ANÁLISE EXPLORATÓRIA DE DADOS--------

# Visualizar as primeiras linhas
print(df.head())

# Estatísticas descritivas
print(df.describe())

# Visualização de distribuição de uma coluna 
sns.histplot(df['product'], bins=30)
plt.show()

# --------MODELAGEM E PREVISÃO--------

# Separar as variáveis independentes e dependentes
X = df[['quantity']]  
y = df['price_y']  # Coluna que quer prever

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# --------SEGMENTAÇÃO E AGRUPAMENTO--------

# Normalizar os dados (opcional, dependendo dos dados)
scaler = StandardScaler()
# Selecione as colunas que você deseja normalizar
X = df[['price_y', 'quantity', 'price_z' ]]
X_scaled = scaler.fit_transform(X)

# Alterar o nome das colunas
df = df.rename(columns={
    "name": "Nome", 
    "sale_id": "id_venda", 
    "product_id": "id_produto", 
    "product": "produto", 
    "price_y": "preço_y", 
    "quantity": "quantidade",
    "price_z": "valor_total", 
    "sale_date": "data_venda"
})

# Salvar o DataFrame renomeado em um novo arquivo CSV
df.to_csv('correctsales.csv', index=False)
