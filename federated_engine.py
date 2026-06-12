"""
=====================================================
Federated Learning Engine - FedAvg Implementation
=====================================================
Implements the Federated Averaging (FedAvg) algorithm
by McMahan et al. (2017) for cross-hospital training.

Key Concepts:
- Each hospital trains locally on its own data
- Only model weights (not data) are shared
- Server aggregates weights using weighted averaging
- Process repeats for multiple rounds
=====================================================
"""

import numpy as np
import copy
import time
from sklearn.model_selection import train_test_split
from ml.model import (HealthcareRiskModel, get_input_dim, train_local_model,
                       get_model_weights, set_model_weights)
from ml.dataset_generator import HealthcareDataGenerator


class FederatedServer:
    """
    Central server that orchestrates federated learning.
    Implements FedAvg aggregation algorithm.
    """

    def __init__(self, dataset_type='heart', num_hospitals=5):
        self.dataset_type = dataset_type
        self.num_hospitals = num_hospitals
        self.input_dim = get_input_dim(dataset_type)
        
        # Initialize global model
        self.global_model = HealthcareRiskModel(self.input_dim)
        self.global_weights = get_model_weights(self.global_model)
        
        # Data generator
        self.data_gen = HealthcareDataGenerator()
        
        # Hospital data stores
        self.hospital_data = {}
        self.hospital_scalers = {}
        
        # Training history
        self.round_history = []
        self.hospital_histories = {i: [] for i in range(num_hospitals)}
        
        # Privacy settings
        self.privacy_budget = 1.0  # Differential privacy epsilon
        self.noise_multiplier = 0.1
        
        print(f"🖥️  Federated Server initialized")
        print(f"   Dataset: {dataset_type} | Hospitals: {num_hospitals}")
        print(f"   Input features: {self.input_dim}")

    def generate_and_distribute_data(self, total_samples=2000):
        """Generate dataset and distribute across hospitals."""
        print(f"\n📊 Generating and distributing data...")
        
        # Generate dataset based on type
        if self.dataset_type == 'heart':
            df = self.data_gen.generate_heart_disease_data(total_samples)
        elif self.dataset_type == 'diabetes':
            df = self.data_gen.generate_diabetes_data(total_samples)
        elif self.dataset_type == 'kidney':
            df = self.data_gen.generate_kidney_disease_data(total_samples)
        else:
            df = self.data_gen.generate_heart_disease_data(total_samples)
        
        # Split across hospitals (non-IID for realistic scenario)
        hospital_dfs = self.data_gen.split_for_hospitals(df, self.num_hospitals, iid=False)
        
        # Preprocess each hospital's data independently
        for h_id, h_df in hospital_dfs.items():
            gen = HealthcareDataGenerator(seed=42 + h_id)
            X, y = gen.preprocess(h_df)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )
            self.hospital_data[h_id] = {
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test,
                'num_samples': len(X_train)
            }
            self.hospital_scalers[h_id] = gen.scaler
        
        return {h: {'num_samples': d['num_samples'], 'test_samples': len(d['X_test'])}
                for h, d in self.hospital_data.items()}

    def add_differential_privacy_noise(self, weights, epsilon=1.0):
        """
        Simulate differential privacy by adding calibrated noise to weights.
        Lower epsilon = more privacy, more noise.
        """
        noisy_weights = {}
        for key, value in weights.items():
            sensitivity = np.max(np.abs(value)) * 0.01
            noise_scale = sensitivity / epsilon
            noise = np.random.laplace(0, noise_scale, value.shape)
            noisy_weights[key] = value + noise.astype(np.float32)
        return noisy_weights

    def federated_averaging(self, hospital_weights, hospital_samples):
        """
        FedAvg Algorithm: Weighted average of model parameters.
        
        w_global = Σ (n_k / n_total) * w_k
        
        where n_k = samples at hospital k, n_total = total samples
        """
        total_samples = sum(hospital_samples.values())
        averaged_weights = {}
        
        for key in hospital_weights[0].keys():
            weighted_sum = np.zeros_like(hospital_weights[0][key])
            for h_id, weights in enumerate(hospital_weights):
                weight_factor = hospital_samples[h_id] / total_samples
                weighted_sum += weights[key] * weight_factor
            averaged_weights[key] = weighted_sum
        
        return averaged_weights

    def train_single_round(self, round_num, local_epochs=5, lr=0.001, batch_size=32):
        """
        Execute one round of federated learning.
        1. Distribute global model to hospitals
        2. Each hospital trains locally
        3. Collect and aggregate updates
        4. Update global model
        """
        print(f"\n{'='*60}")
        print(f"🔄 FEDERATED ROUND {round_num}")
        print(f"{'='*60}")
        
        round_start = time.time()
        hospital_weights_list = []
        hospital_samples = {}
        round_metrics = {}
        
        # Step 1 & 2: Each hospital trains locally
        for h_id in range(self.num_hospitals):
            print(f"\n   🏥 Hospital {h_id} - Training...")
            data = self.hospital_data[h_id]
            
            # Create local model with global weights
            local_model = HealthcareRiskModel(self.input_dim)
            set_model_weights(local_model, copy.deepcopy(self.global_weights))
            
            # Train locally
            local_model, metrics = train_local_model(
                local_model, data['X_train'], data['y_train'],
                data['X_test'], data['y_test'],
                epochs=local_epochs, lr=lr, batch_size=batch_size
            )
            
            # Get updated weights
            local_weights = get_model_weights(local_model)
            
            # Apply differential privacy noise
            if self.privacy_budget < float('inf'):
                local_weights = self.add_differential_privacy_noise(
                    local_weights, self.privacy_budget
                )
            
            hospital_weights_list.append(local_weights)
            hospital_samples[h_id] = data['num_samples']
            round_metrics[h_id] = metrics
            self.hospital_histories[h_id].append(metrics)
        
        # Step 3: Aggregate using FedAvg
        print(f"\n   📡 Aggregating weights using FedAvg...")
        self.global_weights = self.federated_averaging(
            hospital_weights_list, hospital_samples
        )
        
        # Step 4: Update global model
        set_model_weights(self.global_model, self.global_weights)
        
        # Evaluate global model on all test data
        global_metrics = self._evaluate_global_model()
        
        # Calculate communication cost (simulated)
        param_count = sum(p.numel() for p in self.global_model.parameters())
        comm_cost = (param_count * 4 * self.num_hospitals * 2) / (1024 * 1024)  # MB
        
        round_time = time.time() - round_start
        
        round_result = {
            'round': round_num,
            'global_accuracy': global_metrics['accuracy'],
            'global_loss': global_metrics['loss'],
            'global_f1': global_metrics['f1'],
            'global_auc': global_metrics['auc'],
            'hospital_metrics': round_metrics,
            'communication_cost': round(comm_cost, 2),
            'round_time': round(round_time, 2),
            'participating_hospitals': self.num_hospitals
        }
        
        self.round_history.append(round_result)
        
        print(f"\n   📊 Global Accuracy: {global_metrics['accuracy']:.4f}")
        print(f"   📊 Global F1 Score: {global_metrics['f1']:.4f}")
        print(f"   📊 Global AUC: {global_metrics['auc']:.4f}")
        print(f"   ⏱️  Round Time: {round_time:.1f}s")
        print(f"   📡 Communication Cost: {comm_cost:.2f} MB")
        
        return round_result

    def _evaluate_global_model(self):
        """Evaluate the global model on all hospitals' test data."""
        import torch
        self.global_model.eval()
        all_preds = []
        all_true = []
        all_probs = []
        total_loss = 0
        criterion = torch.nn.BCELoss()
        
        with torch.no_grad():
            for h_id in range(self.num_hospitals):
                data = self.hospital_data[h_id]
                X_t = torch.FloatTensor(data['X_test'])
                y_t = torch.FloatTensor(data['y_test'])
                outputs = self.global_model(X_t)
                total_loss += criterion(outputs, y_t).item()
                probs = outputs.numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds)
                all_true.extend(data['y_test'].astype(int))
                all_probs.extend(probs)
        
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
        
        acc = accuracy_score(all_true, all_preds)
        f1 = f1_score(all_true, all_preds, zero_division=0)
        prec = precision_score(all_true, all_preds, zero_division=0)
        rec = recall_score(all_true, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_true, all_probs)
        except ValueError:
            auc = 0.5
        
        return {
            'accuracy': acc, 'loss': total_loss / self.num_hospitals,
            'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc
        }

    def run_federated_training(self, num_rounds=10, local_epochs=5,
                               lr=0.001, batch_size=32):
        """Run the complete federated learning process."""
        print(f"\n{'='*60}")
        print(f"🚀 STARTING FEDERATED LEARNING")
        print(f"   Rounds: {num_rounds} | Local Epochs: {local_epochs}")
        print(f"   Learning Rate: {lr} | Batch Size: {batch_size}")
        print(f"{'='*60}")
        
        if not self.hospital_data:
            self.generate_and_distribute_data()
        
        all_results = []
        for round_num in range(1, num_rounds + 1):
            result = self.train_single_round(round_num, local_epochs, lr, batch_size)
            all_results.append(result)
        
        print(f"\n{'='*60}")
        print(f"✅ FEDERATED LEARNING COMPLETE!")
        print(f"   Final Global Accuracy: {all_results[-1]['global_accuracy']:.4f}")
        print(f"   Final Global F1: {all_results[-1]['global_f1']:.4f}")
        print(f"   Final Global AUC: {all_results[-1]['global_auc']:.4f}")
        print(f"{'='*60}")
        
        return all_results

    def get_training_summary(self):
        """Get a summary of all training rounds."""
        if not self.round_history:
            return None
        return {
            'total_rounds': len(self.round_history),
            'final_accuracy': self.round_history[-1]['global_accuracy'],
            'final_loss': self.round_history[-1]['global_loss'],
            'final_f1': self.round_history[-1]['global_f1'],
            'final_auc': self.round_history[-1]['global_auc'],
            'accuracy_history': [r['global_accuracy'] for r in self.round_history],
            'loss_history': [r['global_loss'] for r in self.round_history],
            'f1_history': [r['global_f1'] for r in self.round_history],
            'auc_history': [r['global_auc'] for r in self.round_history],
            'communication_costs': [r['communication_cost'] for r in self.round_history],
            'round_times': [r['round_time'] for r in self.round_history],
        }
