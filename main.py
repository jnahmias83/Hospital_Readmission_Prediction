import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# -------------------------------
# 1️⃣ Charger et préparer les données
# -------------------------------
df = pd.read_csv('input/diabetic_data.csv')
df.columns = df.columns.str.strip()

# Supprimer les lignes où la cible est manquante
df.dropna(subset=['readmitted'], inplace=True)

# Supprimer les colonnes inutiles
df.drop(['encounter_id','patient_nbr','admission_type_id',
         'discharge_disposition_id','admission_source_id'], axis=1, inplace=True)

# Mapper la variable cible
df['readmitted'] = df['readmitted'].map({'<30':1,'>30':0, 'No':0})

# Copier pour éviter de modifier l'original
df_copy = df.copy()

# -------------------------------
# 2️⃣ Séparer X et y
# -------------------------------
X = df_copy.drop("readmitted", axis=1)
y = df_copy["readmitted"]

# Supprimer les lignes où y est manquant (sécurité)
X = X[y.notna()]
y = y[y.notna()]

# -------------------------------
# 3️⃣ Encoder les colonnes catégorielles
# -------------------------------
cat_cols = X.select_dtypes(include=['object']).columns
X[cat_cols] = X[cat_cols].fillna('Unknown')

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])

# -------------------------------
# 4️⃣ Split train/test et oversampling
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# -------------------------------
# 5️⃣ Random Forest
# -------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_res, y_res)
y_pred = rf.predict(X_test)

# -------------------------------
# 6️⃣ Évaluation
# -------------------------------
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report RF:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, cmap='Blues', annot=True, fmt='d')
plt.title('Matrice de confusion')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.show()

# -------------------------------
# 7️⃣ Importance des features
# -------------------------------
feature_importances = rf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature':features,'Importance':feature_importances})
importance_df = importance_df.sort_values(by='Importance',ascending=False)

plt.figure(figsize=(8,12))
plt.tick_params(axis='x', rotation=0, labelsize=10)
plt.tick_params(axis='y', rotation=0, labelsize=10)
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.tight_layout()
plt.show()

# -------------------------------
# 8️⃣ Graphiques additionnels
# -------------------------------
sns.countplot(data=df, x='readmitted', hue='gender')
plt.title('Count Readmitted by gender')
plt.xlabel('Readmitted')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(14, 10))
numerical_columns = df.select_dtypes(include='number')
sns.heatmap(numerical_columns.corr(), cmap='coolwarm', center=0, annot=True)
plt.title('Correlation Matrix')
plt.show()

# -------------------------------
# 9️⃣ Prédiction pour de nouveaux patients
# -------------------------------
# Exemple de nouveau patient (remplir avec vos valeurs)
new_patients = pd.DataFrame([{
    'race':'Caucasian',
    'gender':'Male',
    'age':'[70-80)',  # Nouvelle valeur possible
    'weight':'[70-80)',
    'time_in_hospital':5,
    'num_lab_procedures':40,
    'num_procedures':0,
    'num_medications':10,
    'number_outpatient':0,
    'number_emergency':0,
    'number_inpatient':0,
    'diag_1':'250.83',
    'diag_2':'401.9',
    'diag_3':'414.01',
    # ajouter toutes les autres colonnes si nécessaire
}])

# Ajouter les colonnes manquantes et réordonner
for col in X_train.columns:
    if col not in new_patients.columns:
        if col in cat_cols:
            new_patients[col] = 'Unknown'
        else:
            new_patients[col] = 0
new_patients = new_patients[X_train.columns]

# Encoder avec le même OrdinalEncoder
new_patients[cat_cols] = encoder.transform(new_patients[cat_cols])

# Prédire probabilité
probas = rf.predict_proba(new_patients)[:, 1]  # probabilité readmitted <30j
print("Probabilité de réadmission <30 jours pour le nouveau patient :", probas)