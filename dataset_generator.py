"""
=====================================================
Healthcare Dataset Generator & Preprocessor
=====================================================
Generates realistic synthetic healthcare datasets for:
- Heart Disease, Diabetes, Chronic Kidney Disease
=====================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class HealthcareDataGenerator:
    """Generates and preprocesses synthetic healthcare datasets."""

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.scaler = StandardScaler()

    def generate_heart_disease_data(self, n_samples=1000):
        """Generate synthetic heart disease dataset (UCI-style)."""
        print(f"🫀 Generating heart disease dataset ({n_samples} samples)...")
        age = np.random.normal(54, 9, n_samples).clip(29, 77).astype(int)
        sex = np.random.binomial(1, 0.68, n_samples)
        cp = np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.17, 0.28, 0.08])
        trestbps = np.random.normal(131, 17, n_samples).clip(94, 200).astype(int)
        chol = np.random.normal(246, 52, n_samples).clip(126, 564).astype(int)
        fbs = np.random.binomial(1, 0.15, n_samples)
        restecg = np.random.choice([0, 1, 2], n_samples, p=[0.49, 0.47, 0.04])
        thalach = np.random.normal(149, 23, n_samples).clip(71, 202).astype(int)
        exang = np.random.binomial(1, 0.33, n_samples)
        oldpeak = np.random.exponential(1.0, n_samples).clip(0, 6.2).round(1)
        slope = np.random.choice([0, 1, 2], n_samples, p=[0.21, 0.46, 0.33])
        ca = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.58, 0.22, 0.13, 0.05, 0.02])
        thal = np.random.choice([0, 1, 2, 3], n_samples, p=[0.02, 0.06, 0.55, 0.37])

        risk_score = (0.03*(age-40) + 0.5*sex + 0.3*(cp==0).astype(int) +
                      0.01*(trestbps-120) + 0.005*(chol-200) + 0.3*fbs +
                      -0.01*(thalach-150) + 0.8*exang + 0.5*oldpeak + 0.3*ca)
        prob = 1 / (1 + np.exp(-risk_score + 2))
        target = (np.random.random(n_samples) < prob).astype(int)

        df = pd.DataFrame({'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca,
            'thal': thal, 'target': target})
        print(f"   ✅ Generated {n_samples} samples | Disease rate: {target.mean():.1%}")
        return df

    def generate_diabetes_data(self, n_samples=1000):
        """Generate synthetic diabetes dataset (Pima-style)."""
        print(f"🩸 Generating diabetes dataset ({n_samples} samples)...")
        pregnancies = np.random.poisson(3.8, n_samples).clip(0, 17)
        glucose = np.random.normal(121, 32, n_samples).clip(44, 199).astype(int)
        blood_pressure = np.random.normal(72, 12, n_samples).clip(24, 122).astype(int)
        skin_thickness = np.random.normal(29, 10, n_samples).clip(7, 99).astype(int)
        insulin = np.random.lognormal(4.6, 0.8, n_samples).clip(14, 846).astype(int)
        bmi = np.random.normal(32, 8, n_samples).clip(18, 67.1).round(1)
        diabetes_pedigree = np.random.exponential(0.47, n_samples).clip(0.08, 2.42).round(3)
        age = np.random.normal(33, 12, n_samples).clip(21, 81).astype(int)

        risk_score = (0.01*glucose + 0.05*bmi + 0.02*age + 0.1*pregnancies +
                      0.5*diabetes_pedigree + 0.001*insulin)
        prob = 1 / (1 + np.exp(-risk_score + 5))
        target = (np.random.random(n_samples) < prob).astype(int)

        df = pd.DataFrame({'pregnancies': pregnancies, 'glucose': glucose,
            'blood_pressure': blood_pressure, 'skin_thickness': skin_thickness,
            'insulin': insulin, 'bmi': bmi, 'diabetes_pedigree': diabetes_pedigree,
            'age': age, 'target': target})
        print(f"   ✅ Generated {n_samples} samples | Diabetes rate: {target.mean():.1%}")
        return df

    def generate_kidney_disease_data(self, n_samples=1000):
        """Generate synthetic CKD dataset (UCI-style)."""
        print(f"🫘 Generating kidney disease dataset ({n_samples} samples)...")
        age = np.random.normal(51, 18, n_samples).clip(2, 90).astype(int)
        bp = np.random.normal(77, 14, n_samples).clip(50, 180).astype(int)
        sg = np.random.choice([1.005,1.010,1.015,1.020,1.025], n_samples, p=[.1,.2,.25,.3,.15])
        albumin = np.random.choice([0,1,2,3,4,5], n_samples, p=[.4,.15,.15,.15,.1,.05])
        sugar = np.random.choice([0,1,2,3,4,5], n_samples, p=[.5,.1,.15,.1,.1,.05])
        rbc = np.random.binomial(1, 0.38, n_samples)
        pus = np.random.binomial(1, 0.25, n_samples)
        bacteria = np.random.binomial(1, 0.12, n_samples)
        bgr = np.random.lognormal(4.8, 0.5, n_samples).clip(22, 490).astype(int)
        bu = np.random.lognormal(3.5, 0.7, n_samples).clip(1.5, 391).round(1)
        sc = np.random.lognormal(0.8, 0.8, n_samples).clip(0.4, 76).round(1)
        sodium = np.random.normal(137, 10, n_samples).clip(4.5, 163).round(1)
        potassium = np.random.normal(4.6, 2, n_samples).clip(2.5, 47).round(1)
        hemo = np.random.normal(12.5, 3, n_samples).clip(3.1, 17.8).round(1)
        pcv = np.random.normal(38, 9, n_samples).clip(9, 54).astype(int)
        wc = np.random.lognormal(8.8, 0.4, n_samples).clip(2200, 26400).astype(int)
        rc = np.random.normal(4.7, 1.1, n_samples).clip(2.1, 8.0).round(1)
        htn = np.random.binomial(1, 0.38, n_samples)
        dm = np.random.binomial(1, 0.31, n_samples)

        risk_score = (0.02*(age-30) + 0.5*albumin + 0.3*sugar + 0.8*rbc + 0.5*pus +
                      0.3*bacteria + 0.01*(bgr-100) + 0.02*bu + 0.5*(sc-1) -
                      0.1*(hemo-12) + 0.5*htn + 0.3*dm)
        prob = 1 / (1 + np.exp(-risk_score + 3))
        target = (np.random.random(n_samples) < prob).astype(int)

        df = pd.DataFrame({'age': age, 'blood_pressure': bp, 'specific_gravity': sg,
            'albumin': albumin, 'sugar': sugar, 'red_blood_cells': rbc,
            'pus_cell': pus, 'bacteria': bacteria, 'blood_glucose': bgr,
            'blood_urea': bu, 'serum_creatinine': sc, 'sodium': sodium,
            'potassium': potassium, 'hemoglobin': hemo, 'packed_cell_volume': pcv,
            'white_blood_cell_count': wc, 'red_blood_cell_count': rc,
            'hypertension': htn, 'diabetes_mellitus': dm, 'target': target})
        print(f"   ✅ Generated {n_samples} samples | CKD rate: {target.mean():.1%}")
        return df

    def split_for_hospitals(self, df, num_hospitals=5, iid=False):
        """Split dataset across hospitals for federated learning simulation."""
        hospital_data = {}
        if iid:
            indices = np.random.permutation(len(df))
            splits = np.array_split(indices, num_hospitals)
            for i, s in enumerate(splits):
                hospital_data[i] = df.iloc[s].reset_index(drop=True)
        else:
            total = len(df)
            proportions = np.random.dirichlet(np.ones(num_hospitals) * 2)
            sizes = (proportions * total).astype(int)
            sizes[-1] = total - sizes[:-1].sum()
            indices = np.random.permutation(len(df))
            start = 0
            for i, size in enumerate(sizes):
                hospital_data[i] = df.iloc[indices[start:start+size]].reset_index(drop=True)
                start += size
        for i, data in hospital_data.items():
            print(f"   🏥 Hospital {i}: {len(data)} samples | Positive: {data['target'].mean():.1%}")
        return hospital_data

    def preprocess(self, df, fit_scaler=True):
        """Preprocess dataset: separate features/target, handle missing values, scale."""
        X = df.drop('target', axis=1).values.astype(np.float32)
        y = df['target'].values.astype(np.float32)
        col_means = np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X[mask, i] = col_means[i]
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        return X, y

    def get_feature_names(self, dataset_type):
        features = {
            'heart': ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'],
            'diabetes': ['pregnancies','glucose','blood_pressure','skin_thickness','insulin','bmi','diabetes_pedigree','age'],
            'kidney': ['age','blood_pressure','specific_gravity','albumin','sugar','red_blood_cells','pus_cell','bacteria','blood_glucose','blood_urea','serum_creatinine','sodium','potassium','hemoglobin','packed_cell_volume','white_blood_cell_count','red_blood_cell_count','hypertension','diabetes_mellitus']
        }
        return features.get(dataset_type, [])

    def get_num_features(self, dataset_type):
        return len(self.get_feature_names(dataset_type))
