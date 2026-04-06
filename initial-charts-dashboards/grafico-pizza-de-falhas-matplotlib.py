import matplotlib.pyplot as plt

causas = ['Mecânica', 'Elétrica', 'Logística', 'Clima', 'Sensores']
frequencia = [40, 25, 15, 10, 10]

plt.pie(frequencia, labels=causas, autopct='%1.1f%%', startangle=140, explode=(0.1, 0, 0, 0, 0))
plt.title("Causas de Paradas (Non-Productive Time)")
plt.show()
