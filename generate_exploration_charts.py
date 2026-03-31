import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from bank_patron_losing import BankDataPreprocessor

os.makedirs('./report_images', exist_ok=True)
df = pd.read_csv('./dataset/Churn-Modelling-0-original.csv')

# ============ 1. 数据概览信息图 ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Data Exploration Overview', fontsize=16, fontweight='bold')

# 1.1 类别分布
ax1 = axes[0, 0]
exited_counts = df['Exited'].value_counts()
colors_exited = ['#5CB85C', '#E74C3C']
bars1 = ax1.bar(['Retained (0)', 'Churned (1)'], exited_counts.values, color=colors_exited, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Target Variable Distribution (Class Imbalance)', fontsize=13, fontweight='bold')
ax1.text(0, exited_counts.values[0] + 200, f'{exited_counts.values[0]}\n({exited_counts.values[0]/len(df)*100:.1f}%)',
         ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.text(1, exited_counts.values[1] + 200, f'{exited_counts.values[1]}\n({exited_counts.values[1]/len(df)*100:.1f}%)',
         ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.axhline(y=exited_counts.values.mean(), color='gray', linestyle='--', alpha=0.5, label='Average')

# 1.2 地理分布
ax2 = axes[0, 1]
geo_churn = df.groupby('Geography')['Exited'].agg(['count', 'sum'])
geo_churn['rate'] = geo_churn['sum'] / geo_churn['count'] * 100
colors_geo = ['#3498db', '#e74c3c', '#2ecc71']
bars2 = ax2.bar(geo_churn.index, geo_churn['rate'], color=colors_geo, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Churn Rate (%)', fontsize=12)
ax2.set_title('Churn Rate by Geography', fontsize=13, fontweight='bold')
ax2.axhline(y=df['Exited'].mean()*100, color='gray', linestyle='--', alpha=0.5, label=f'Average: {df["Exited"].mean()*100:.1f}%')
for i, (idx, row) in enumerate(geo_churn.iterrows()):
    ax2.text(i, row['rate'] + 1, f'{row["rate"]:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.legend()

# 1.3 性别分布
ax3 = axes[1, 0]
gender_churn = df.groupby('Gender')['Exited'].agg(['count', 'sum'])
gender_churn['rate'] = gender_churn['sum'] / gender_churn['count'] * 100
colors_gender = ['#9b59b6', '#3498db']
bars3 = ax3.bar(gender_churn.index, gender_churn['rate'], color=colors_gender, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Churn Rate (%)', fontsize=12)
ax3.set_title('Churn Rate by Gender', fontsize=13, fontweight='bold')
ax3.axhline(y=df['Exited'].mean()*100, color='gray', linestyle='--', alpha=0.5)
for i, (idx, row) in enumerate(gender_churn.iterrows()):
    ax3.text(i, row['rate'] + 1, f'{row["rate"]:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 1.4 年龄段分布
ax4 = axes[1, 1]
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '>60'])
age_churn = df.groupby('AgeGroup')['Exited'].agg(['count', 'sum'])
age_churn['rate'] = age_churn['sum'] / age_churn['count'] * 100
colors_age = ['#27ae60', '#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
bars4 = ax4.bar(age_churn.index, age_churn['rate'], color=colors_age, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Churn Rate (%)', fontsize=12)
ax4.set_title('Churn Rate by Age Group', fontsize=13, fontweight='bold')
ax4.axhline(y=df['Exited'].mean()*100, color='gray', linestyle='--', alpha=0.5)
for i, (idx, row) in enumerate(age_churn.iterrows()):
    ax4.text(i, row['rate'] + 1, f'{row["rate"]:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=0)

plt.tight_layout()
plt.savefig('./report_images/data_exploration.png', dpi=150, bbox_inches='tight')
plt.close()
print('已保存：data_exploration.png')

# ============ 2. 特征分布直方图 ============
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Feature Distribution Analysis', fontsize=16, fontweight='bold')

features_to_plot = ['CreditScore', 'Age', 'Balance', 'Tenure', 'NumOfProducts', 'EstimatedSalary']
for idx, feature in enumerate(features_to_plot):
    ax = axes[idx // 3, idx % 3]

    churned = df[df['Exited'] == 1][feature]
    retained = df[df['Exited'] == 0][feature]

    ax.hist(retained, bins=30, alpha=0.5, label='Retained', color='#5CB85C', density=True)
    ax.hist(churned, bins=30, alpha=0.5, label='Churned', color='#E74C3C', density=True)

    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('./report_images/feature_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('已保存：feature_distribution.png')

# ============ 3. 预处理前后对比图 ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Data Preprocessing Comparison', fontsize=16, fontweight='bold')

# 3.1 原始数据 Age 分布
ax1 = axes[0, 0]
ax1.hist(df['Age'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Age')
ax1.set_ylabel('Count')
ax1.set_title('Original Age Distribution (Raw Data)', fontsize=13, fontweight='bold')
ax1.axvline(x=df['Age'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Age"].mean():.1f}')
ax1.axvline(x=df['Age'].std() + df['Age'].mean(), color='orange', linestyle=':', linewidth=2, label=f'+1 Std: {df["Age"].mean() + df["Age"].std():.1f}')
ax1.legend()

# 3.2 标准化后 Age 分布
ax2 = axes[0, 1]
scaler = StandardScaler()
age_scaled = scaler.fit_transform(df[['Age']])
ax2.hist(age_scaled, bins=30, color='#27ae60', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Age (Standardized)')
ax2.set_ylabel('Count')
ax2.set_title('Standardized Age Distribution (Z-score)', fontsize=13, fontweight='bold')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Mean: 0')
ax2.axvline(x=1, color='orange', linestyle=':', linewidth=2, label='+1 Std')
ax2.axvline(x=-1, color='orange', linestyle=':', linewidth=2)
ax2.legend()

# 3.3 离散化前后对比
ax3 = axes[1, 0]
preprocessor_dt = BankDataPreprocessor(random_state=10)
preprocessor_dt.fit(df, discretize=True)
q1 = preprocessor_dt.quantiles['Age']['q1']
q2 = preprocessor_dt.quantiles['Age']['q2']

ax3.hist(df['Age'], bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
ax3.axvline(x=q1, color='red', linestyle='--', linewidth=2, label=f'33% Quantile: {q1:.1f}')
ax3.axvline(x=q2, color='orange', linestyle='--', linewidth=2, label=f'66% Quantile: {q2:.1f}')
ax3.set_xlabel('Age')
ax3.set_ylabel('Count')
ax3.set_title('Age Discretization (Quantile Binning)', fontsize=13, fontweight='bold')
ax3.legend()

# 3.4 类别平衡对比
ax4 = axes[1, 1]
before = [7963, 2037]
after = [1630, 1629]
x = np.array([0, 1])
width = 0.35
ax4.bar(x - width/2, before, width, label='Before Balancing', color='#e74c3c', edgecolor='black')
ax4.bar(x + width/2, after, width, label='After Balancing', color='#27ae60', edgecolor='black')
ax4.set_xticks(x)
ax4.set_xticklabels(['Retained (0)', 'Churned (1)'])
ax4.set_ylabel('Count')
ax4.set_title('Class Balance Comparison', fontsize=13, fontweight='bold')
ax4.legend()

plt.tight_layout()
plt.savefig('./report_images/preprocessing_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('已保存：preprocessing_comparison.png')

print('\n所有探索性分析图片已保存!')
