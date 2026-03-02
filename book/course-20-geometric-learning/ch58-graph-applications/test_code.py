#!/usr/bin/env python3
"""
Code Review Test Script for Chapter 58
Tests all code blocks in sequence
"""

import sys
import traceback

def test_part_1():
    """Test Part 1: Molecular Property Prediction with Message Passing"""
    print("\n" + "="*80)
    print("Testing Part 1: Molecular Property Prediction Dataset")
    print("="*80)

    try:
        # Molecular Property Prediction: ESOL Solubility Dataset
        import torch
        import torch.nn.functional as F
        from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
        from torch_geometric.datasets import MoleculeNet
        from torch_geometric.nn import GCNConv, global_mean_pool
        from torch_geometric.loader import DataLoader
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import numpy as np
        import matplotlib.pyplot as plt

        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Load ESOL dataset (water solubility)
        dataset = MoleculeNet(root='./data', name='ESOL')

        print(f"Dataset: {dataset}")
        print(f"Number of molecules: {len(dataset)}")
        print(f"Number of node features: {dataset.num_node_features}")
        print(f"Number of edge features: {dataset.num_edge_features}")

        # Examine a molecule
        mol = dataset[0]
        print(f"\nExample molecule:")
        print(f"  Nodes (atoms): {mol.num_nodes}")
        print(f"  Edges (bonds): {mol.num_edges}")
        print(f"  Node features shape: {mol.x.shape}")
        print(f"  Solubility (log mol/L): {mol.y.item():.3f}")

        print("✓ Part 1 passed")
        return True

    except Exception as e:
        print(f"✗ Part 1 failed: {e}")
        traceback.print_exc()
        return False


def test_part_2():
    """Test Part 2: Building a Message Passing Neural Network"""
    print("\n" + "="*80)
    print("Testing Part 2: Building MPNN")
    print("="*80)

    try:
        import torch
        import torch.nn.functional as F
        from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
        from torch_geometric.datasets import MoleculeNet
        from torch_geometric.nn import GCNConv, global_mean_pool
        import numpy as np

        torch.manual_seed(42)
        np.random.seed(42)

        dataset = MoleculeNet(root='./data', name='ESOL')

        # Define Message Passing Neural Network for molecules
        class MolecularGNN(torch.nn.Module):
            def __init__(self, num_node_features, hidden_dim=64):
                super(MolecularGNN, self).__init__()

                # Message passing layers
                self.conv1 = GCNConv(num_node_features, hidden_dim)
                self.bn1 = BatchNorm1d(hidden_dim)

                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.bn2 = BatchNorm1d(hidden_dim)

                self.conv3 = GCNConv(hidden_dim, hidden_dim)
                self.bn3 = BatchNorm1d(hidden_dim)

                # Graph-level prediction head
                self.mlp = Sequential(
                    Linear(hidden_dim, hidden_dim // 2),
                    ReLU(),
                    Linear(hidden_dim // 2, 1)
                )

            def forward(self, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch

                # Message passing: aggregate neighbor information
                x = self.conv1(x, edge_index)
                x = self.bn1(x)
                x = F.relu(x)

                x = self.conv2(x, edge_index)
                x = self.bn2(x)
                x = F.relu(x)

                x = self.conv3(x, edge_index)
                x = self.bn3(x)
                x = F.relu(x)

                # Global pooling: aggregate node embeddings to graph embedding
                x = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]

                # Predict molecular property
                x = self.mlp(x)
                return x

        model = MolecularGNN(num_node_features=dataset.num_node_features)
        print(model)
        print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

        print("✓ Part 2 passed")
        return True

    except Exception as e:
        print(f"✗ Part 2 failed: {e}")
        traceback.print_exc()
        return False


def test_part_3():
    """Test Part 3: Training and Evaluation"""
    print("\n" + "="*80)
    print("Testing Part 3: Training and Evaluation (reduced epochs for speed)")
    print("="*80)

    try:
        import torch
        import torch.nn.functional as F
        from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
        from torch_geometric.datasets import MoleculeNet
        from torch_geometric.nn import GCNConv, global_mean_pool
        from torch_geometric.loader import DataLoader
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import numpy as np

        torch.manual_seed(42)
        np.random.seed(42)

        dataset = MoleculeNet(root='./data', name='ESOL')

        class MolecularGNN(torch.nn.Module):
            def __init__(self, num_node_features, hidden_dim=64):
                super(MolecularGNN, self).__init__()
                self.conv1 = GCNConv(num_node_features, hidden_dim)
                self.bn1 = BatchNorm1d(hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.bn2 = BatchNorm1d(hidden_dim)
                self.conv3 = GCNConv(hidden_dim, hidden_dim)
                self.bn3 = BatchNorm1d(hidden_dim)
                self.mlp = Sequential(
                    Linear(hidden_dim, hidden_dim // 2),
                    ReLU(),
                    Linear(hidden_dim // 2, 1)
                )

            def forward(self, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch
                x = self.conv1(x, edge_index)
                x = self.bn1(x)
                x = F.relu(x)
                x = self.conv2(x, edge_index)
                x = self.bn2(x)
                x = F.relu(x)
                x = self.conv3(x, edge_index)
                x = self.bn3(x)
                x = F.relu(x)
                x = global_mean_pool(x, batch)
                x = self.mlp(x)
                return x

        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.random.manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)

        # Training function
        def train_epoch(model, loader, optimizer):
            model.train()
            total_loss = 0
            for data in loader:
                optimizer.zero_grad()
                out = model(data)
                loss = F.mse_loss(out.squeeze(), data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.num_graphs
            return total_loss / len(loader.dataset)

        def evaluate(model, loader):
            model.eval()
            predictions, targets = [], []
            with torch.no_grad():
                for data in loader:
                    out = model(data)
                    predictions.append(out.squeeze())
                    targets.append(data.y)
            predictions = torch.cat(predictions).numpy()
            targets = torch.cat(targets).numpy()
            rmse = np.sqrt(mean_squared_error(targets, predictions))
            mae = mean_absolute_error(targets, predictions)
            return rmse, mae, predictions, targets

        # Train GNN (reduced epochs for testing)
        model = MolecularGNN(num_node_features=dataset.num_node_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        train_losses, val_rmses = [], []

        print("Training Molecular GNN...")
        for epoch in range(1, 21):  # Reduced from 101 to 21
            train_loss = train_epoch(model, train_loader, optimizer)
            val_rmse, val_mae, _, _ = evaluate(model, val_loader)
            train_losses.append(train_loss)
            val_rmses.append(val_rmse)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                      f"Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f}")

        # Final test evaluation
        test_rmse, test_mae, test_preds, test_targets = evaluate(model, test_loader)
        print(f"\nGNN Test Results:")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE:  {test_mae:.4f}")

        print("✓ Part 3 passed")
        return True, model, test_rmse, test_preds, test_targets

    except Exception as e:
        print(f"✗ Part 3 failed: {e}")
        traceback.print_exc()
        return False, None, None, None, None


def test_part_4(test_rmse_gnn=None, test_preds_gnn=None, test_targets_gnn=None):
    """Test Part 4: Baseline Comparison"""
    print("\n" + "="*80)
    print("Testing Part 4: Baseline Comparison")
    print("="*80)

    try:
        import torch
        import numpy as np
        from torch_geometric.datasets import MoleculeNet
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import pandas as pd
        import matplotlib.pyplot as plt

        torch.manual_seed(42)
        np.random.seed(42)

        dataset = MoleculeNet(root='./data', name='ESOL')

        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.random.manual_seed(42)
        )

        # Create baseline: Random Forest on Morgan fingerprints
        def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
            """Convert SMILES string to Morgan fingerprint (tabular features)."""
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(n_bits)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return np.array(fp)

        # Extract SMILES from dataset (stored in raw data)
        df = pd.read_csv('./data/ESOL/raw/delaney-processed.csv')
        smiles_list = df['smiles'].tolist()
        y_all = torch.cat([dataset[i].y for i in range(len(dataset))]).numpy()

        # Create fingerprints for all molecules
        X_fingerprints = np.array([smiles_to_fingerprint(smiles) for smiles in smiles_list])

        # Use same train/val/test split as GNN
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices
        test_indices = test_dataset.indices

        X_train_fp = X_fingerprints[train_indices]
        y_train_fp = y_all[train_indices]
        X_test_fp = X_fingerprints[test_indices]
        y_test_fp = y_all[test_indices]

        # Train Random Forest baseline
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
        rf_model.fit(X_train_fp, y_train_fp)

        # Evaluate baseline
        rf_preds = rf_model.predict(X_test_fp)
        rf_rmse = np.sqrt(mean_squared_error(y_test_fp, rf_preds))
        rf_mae = mean_absolute_error(y_test_fp, rf_preds)

        print("\nRandom Forest (Morgan Fingerprints) Test Results:")
        print(f"  RMSE: {rf_rmse:.4f}")
        print(f"  MAE:  {rf_mae:.4f}")

        if test_rmse_gnn is not None:
            print("\nComparison:")
            print(f"  GNN RMSE: {test_rmse_gnn:.4f}")
            print(f"  RF RMSE:  {rf_rmse:.4f}")
            improvement = ((rf_rmse - test_rmse_gnn) / rf_rmse * 100)
            print(f"  Improvement: {improvement:.1f}%")

        # Visualize predictions (only if we have GNN predictions)
        if test_preds_gnn is not None and test_targets_gnn is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].scatter(test_targets_gnn, test_preds_gnn, alpha=0.6, s=30)
            axes[0].plot([-12, 2], [-12, 2], 'r--', lw=2, label='Perfect prediction')
            axes[0].set_xlabel('True Solubility (log mol/L)')
            axes[0].set_ylabel('Predicted Solubility')
            axes[0].set_title(f'GNN Predictions (RMSE={test_rmse_gnn:.3f})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].scatter(y_test_fp, rf_preds, alpha=0.6, s=30, color='orange')
            axes[1].plot([-12, 2], [-12, 2], 'r--', lw=2, label='Perfect prediction')
            axes[1].set_xlabel('True Solubility (log mol/L)')
            axes[1].set_ylabel('Predicted Solubility')
            axes[1].set_title(f'Random Forest Predictions (RMSE={rf_rmse:.3f})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('molecular_predictions.png', dpi=150, bbox_inches='tight')
            print("\nSaved visualization to 'molecular_predictions.png'")

        print("✓ Part 4 passed")
        return True

    except Exception as e:
        print(f"✗ Part 4 failed: {e}")
        traceback.print_exc()
        return False


def test_part_5():
    """Test Part 5: Fraud Detection with Heterogeneous Graphs"""
    print("\n" + "="*80)
    print("Testing Part 5: Fraud Detection Dataset")
    print("="*80)

    try:
        import torch
        import numpy as np
        import networkx as nx
        from torch_geometric.utils import from_networkx

        # Create synthetic fraud detection dataset
        np.random.seed(42)
        torch.manual_seed(42)

        # Generate heterogeneous graph: users, merchants, devices
        n_users = 500
        n_merchants = 50
        n_devices = 200
        n_transactions = 2000

        # User features: account age, transaction count, average amount
        user_features = np.random.randn(n_users, 5)

        # Merchant features: category, reputation, volume
        merchant_features = np.random.randn(n_merchants, 5)

        # Device features: type, location hash
        device_features = np.random.randn(n_devices, 3)

        # Transaction edges: user -> merchant (with device)
        transactions = []
        fraud_labels = np.zeros(n_users)  # Node classification task

        # Normal transactions
        for _ in range(1800):
            user = np.random.randint(0, n_users)
            merchant = np.random.randint(0, n_merchants)
            device = np.random.randint(0, n_devices)
            transactions.append({
                'user': user,
                'merchant': merchant,
                'device': device,
                'amount': np.abs(np.random.randn() * 100 + 50)
            })

        # Fraud rings: 5 rings of 4 users each, circular transfers
        n_fraud_rings = 5
        fraud_users = set()
        for ring_id in range(n_fraud_rings):
            ring_users = np.random.choice(n_users, size=4, replace=False)
            fraud_users.update(ring_users)
            fraud_labels[ring_users] = 1

            # Create circular transfers within ring
            for i in range(len(ring_users)):
                user = ring_users[i]
                next_user = ring_users[(i + 1) % len(ring_users)]
                # Fraudsters use same device
                device = n_devices - 1 - ring_id

                # Transfer to merchant, then merchant "transfers" to next user
                for _ in range(3):
                    merchant = np.random.randint(0, 5)  # Same small set of merchants
                    transactions.append({
                        'user': user,
                        'merchant': merchant,
                        'device': device,
                        'amount': np.random.uniform(800, 1200)  # High amounts
                    })

        print(f"Total users: {n_users}")
        print(f"Fraudulent users: {int(fraud_labels.sum())} ({fraud_labels.mean()*100:.1f}%)")
        print(f"Total transactions: {len(transactions)}")
        print(f"Class imbalance ratio: {(1 - fraud_labels.mean()) / fraud_labels.mean():.1f}:1")

        # Build heterogeneous graph
        # For simplicity, project to homogeneous user-user graph via shared merchants/devices
        user_graph = np.zeros((n_users, n_users))
        for txn in transactions:
            u1 = txn['user']
            # Connect users who use same device or merchant
            for txn2 in transactions:
                u2 = txn2['user']
                if u1 != u2 and (txn['device'] == txn2['device'] or txn['merchant'] == txn2['merchant']):
                    user_graph[u1, u2] = 1

        # Convert to PyTorch Geometric format
        G = nx.from_numpy_array(user_graph)
        for i in range(n_users):
            G.nodes[i]['x'] = torch.tensor(user_features[i], dtype=torch.float)
            G.nodes[i]['y'] = int(fraud_labels[i])

        data = from_networkx(G, group_node_attrs=['x', 'y'])
        data.x = data.x.float()
        data.y = data.y.long()

        print(f"\nGraph statistics:")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.num_edges}")
        print(f"  Average degree: {data.num_edges / data.num_nodes:.1f}")

        print("✓ Part 5 passed")
        return True, data

    except Exception as e:
        print(f"✗ Part 5 failed: {e}")
        traceback.print_exc()
        return False, None


def test_part_6(data=None):
    """Test Part 6: Fraud Detection Model"""
    print("\n" + "="*80)
    print("Testing Part 6: Fraud Detection Model (reduced epochs)")
    print("="*80)

    try:
        import torch
        import torch.nn.functional as F
        import numpy as np
        from torch_geometric.nn import GCNConv
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        if data is None:
            # Recreate data if not provided
            test_result, data = test_part_5()
            if not test_result:
                raise Exception("Failed to create fraud detection data")

        torch.manual_seed(42)
        np.random.seed(42)

        # GNN for fraud detection with focal loss
        class FraudDetectionGNN(torch.nn.Module):
            def __init__(self, num_features, hidden_dim=32):
                super(FraudDetectionGNN, self).__init__()
                self.conv1 = GCNConv(num_features, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.conv3 = GCNConv(hidden_dim, 2)  # Binary classification

            def forward(self, x, edge_index):
                x = F.relu(self.conv1(x, edge_index))
                x = F.dropout(x, p=0.5, training=self.training)
                x = F.relu(self.conv2(x, edge_index))
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv3(x, edge_index)
                return F.log_softmax(x, dim=1)

        # Focal loss for class imbalance
        def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
            """Focal loss down-weights easy examples, focuses on hard negatives."""
            ce_loss = F.cross_entropy(logits, labels, reduction='none')
            p = torch.exp(-ce_loss)
            focal_weight = (1 - p) ** gamma
            loss = alpha * focal_weight * ce_loss
            return loss.mean()

        # Split data
        num_train = int(0.8 * data.num_nodes)
        num_val = int(0.1 * data.num_nodes)

        indices = torch.randperm(data.num_nodes, generator=torch.Generator().manual_seed(42))
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

        train_mask[indices[:num_train]] = True
        val_mask[indices[num_train:num_train+num_val]] = True
        test_mask[indices[num_train+num_val:]] = True

        # Train fraud detection model
        fraud_model = FraudDetectionGNN(num_features=data.num_features)
        optimizer = torch.optim.Adam(fraud_model.parameters(), lr=0.01, weight_decay=5e-4)

        def train_fraud_epoch():
            fraud_model.train()
            optimizer.zero_grad()
            out = fraud_model(data.x, data.edge_index)
            loss = focal_loss(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()
            return loss.item()

        def evaluate_fraud(mask):
            fraud_model.eval()
            with torch.no_grad():
                logits = fraud_model(data.x, data.edge_index)
                pred = logits[mask].argmax(dim=1)
                correct = (pred == data.y[mask]).sum().item()
                total = mask.sum().item()

                # Compute precision, recall, F1 for fraud class
                tp = ((pred == 1) & (data.y[mask] == 1)).sum().item()
                fp = ((pred == 1) & (data.y[mask] == 0)).sum().item()
                fn = ((pred == 0) & (data.y[mask] == 1)).sum().item()

                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)

            return correct / total, precision, recall, f1

        print("Training Fraud Detection GNN...")
        for epoch in range(1, 51):  # Reduced from 201 to 51
            train_loss = train_fraud_epoch()

            if epoch % 25 == 0:
                train_acc, train_prec, train_rec, train_f1 = evaluate_fraud(train_mask)
                val_acc, val_prec, val_rec, val_f1 = evaluate_fraud(val_mask)
                print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                      f"Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f}")

        # Final evaluation
        test_acc, test_prec, test_rec, test_f1 = evaluate_fraud(test_mask)
        print(f"\nTest Results:")
        print(f"  Accuracy:  {test_acc:.3f}")
        print(f"  Precision: {test_prec:.3f}")
        print(f"  Recall:    {test_rec:.3f}")
        print(f"  F1 Score:  {test_f1:.3f}")

        # Baseline: Logistic Regression on features only (no graph)
        X_features = data.x.numpy()
        y_labels = data.y.numpy()

        lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        lr_model.fit(X_features[train_mask.numpy()], y_labels[train_mask.numpy()])

        lr_pred = lr_model.predict(X_features[test_mask.numpy()])
        lr_acc = accuracy_score(y_labels[test_mask.numpy()], lr_pred)
        lr_prec = precision_score(y_labels[test_mask.numpy()], lr_pred, zero_division=0)
        lr_rec = recall_score(y_labels[test_mask.numpy()], lr_pred, zero_division=0)
        lr_f1 = f1_score(y_labels[test_mask.numpy()], lr_pred, zero_division=0)

        print(f"\nLogistic Regression (No Graph) Test Results:")
        print(f"  Accuracy:  {lr_acc:.3f}")
        print(f"  Precision: {lr_prec:.3f}")
        print(f"  Recall:    {lr_rec:.3f}")
        print(f"  F1 Score:  {lr_f1:.3f}")

        improvement = ((test_f1 - lr_f1) / (lr_f1 + 1e-10) * 100)
        print(f"\nF1 Improvement: {improvement:.1f}%")

        print("✓ Part 6 passed")
        return True

    except Exception as e:
        print(f"✗ Part 6 failed: {e}")
        traceback.print_exc()
        return False


def test_part_7():
    """Test Part 7: Knowledge Graph Embeddings with TransE"""
    print("\n" + "="*80)
    print("Testing Part 7: TransE Knowledge Graph Embeddings (reduced epochs)")
    print("="*80)

    try:
        import torch
        import torch.nn.functional as F
        import numpy as np

        torch.manual_seed(42)
        np.random.seed(42)

        # Knowledge Graph Embeddings: TransE implementation
        # Using a subset of FB15k-237 facts (simplified for demonstration)

        # Create synthetic knowledge graph triples
        entities = ['Paris', 'France', 'Berlin', 'Germany', 'Rome', 'Italy',
                    'London', 'UK', 'Madrid', 'Spain']
        relations = ['capital_of', 'located_in', 'neighbor_of']

        # Ground truth triples (head, relation, tail)
        triples = [
            ('Paris', 'capital_of', 'France'),
            ('Berlin', 'capital_of', 'Germany'),
            ('Rome', 'capital_of', 'Italy'),
            ('London', 'capital_of', 'UK'),
            ('Madrid', 'capital_of', 'Spain'),
            ('France', 'neighbor_of', 'Germany'),
            ('Germany', 'neighbor_of', 'France'),
            ('France', 'neighbor_of', 'Italy'),
            ('Italy', 'neighbor_of', 'France'),
            ('France', 'neighbor_of', 'Spain'),
            ('Spain', 'neighbor_of', 'France'),
        ]

        # Create mappings
        entity2id = {e: i for i, e in enumerate(entities)}
        relation2id = {r: i for i, r in enumerate(relations)}
        n_entities = len(entities)
        n_relations = len(relations)

        # Convert triples to indices
        train_triples = torch.tensor(
            [[entity2id[h], relation2id[r], entity2id[t]] for h, r, t in triples],
            dtype=torch.long
        )

        print(f"Knowledge Graph:")
        print(f"  Entities: {n_entities}")
        print(f"  Relations: {n_relations}")
        print(f"  Triples: {len(train_triples)}")
        print(f"\nExample triples:")
        for i in range(min(5, len(triples))):
            print(f"  {triples[i]}")

        # TransE model: learns h + r ≈ t
        class TransE(torch.nn.Module):
            def __init__(self, n_entities, n_relations, embedding_dim=50, margin=1.0):
                super(TransE, self).__init__()
                self.entity_embeddings = torch.nn.Embedding(n_entities, embedding_dim)
                self.relation_embeddings = torch.nn.Embedding(n_relations, embedding_dim)
                self.margin = margin

                # Initialize with normalized embeddings
                torch.nn.init.xavier_uniform_(self.entity_embeddings.weight)
                torch.nn.init.xavier_uniform_(self.relation_embeddings.weight)

            def forward(self, triples):
                """Compute score for triples: ||h + r - t||"""
                heads = self.entity_embeddings(triples[:, 0])
                relations = self.relation_embeddings(triples[:, 1])
                tails = self.entity_embeddings(triples[:, 2])

                # L2 norm of (h + r - t)
                scores = torch.norm(heads + relations - tails, p=2, dim=1)
                return scores

            def normalize_embeddings(self):
                """Normalize entity embeddings to unit sphere."""
                self.entity_embeddings.weight.data = F.normalize(
                    self.entity_embeddings.weight.data, p=2, dim=1
                )

        # Create negative samples by corrupting triples
        def generate_negative_samples(positive_triples, n_entities, n_samples=5):
            """Replace head or tail with random entity."""
            negatives = []
            for _ in range(n_samples):
                for triple in positive_triples:
                    corrupted = triple.clone()
                    if np.random.rand() < 0.5:
                        # Corrupt head
                        corrupted[0] = np.random.randint(0, n_entities)
                    else:
                        # Corrupt tail
                        corrupted[2] = np.random.randint(0, n_entities)
                    negatives.append(corrupted)
            return torch.stack(negatives)

        # Train TransE
        transe_model = TransE(n_entities, n_relations, embedding_dim=20)
        optimizer = torch.optim.Adam(transe_model.parameters(), lr=0.01)

        print("\nTraining TransE...")
        for epoch in range(1, 101):  # Reduced from 501 to 101
            transe_model.train()
            optimizer.zero_grad()

            # Positive samples
            pos_scores = transe_model(train_triples)

            # Negative samples
            neg_triples = generate_negative_samples(train_triples, n_entities, n_samples=2)
            neg_scores = transe_model(neg_triples)

            # Margin ranking loss: push positive scores down, negative scores up
            loss = torch.relu(transe_model.margin + pos_scores.mean() - neg_scores.mean())

            loss.backward()
            optimizer.step()
            transe_model.normalize_embeddings()

            if epoch % 50 == 0:
                print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                      f"Pos Score: {pos_scores.mean().item():.3f} | "
                      f"Neg Score: {neg_scores.mean().item():.3f}")

        print("✓ Part 7 passed")
        return True, transe_model, entity2id, relation2id, triples

    except Exception as e:
        print(f"✗ Part 7 failed: {e}")
        traceback.print_exc()
        return False, None, None, None, None


def test_part_8(transe_model=None, entity2id=None, relation2id=None, triples=None):
    """Test Part 8: Link Prediction with TransE"""
    print("\n" + "="*80)
    print("Testing Part 8: Link Prediction with TransE")
    print("="*80)

    try:
        import torch
        import numpy as np

        if transe_model is None:
            # Recreate if not provided
            test_result, transe_model, entity2id, relation2id, triples = test_part_7()
            if not test_result:
                raise Exception("Failed to create TransE model")

        # Test link prediction: predict missing tails given (head, relation)
        test_queries = [
            ('Paris', 'capital_of', '?'),  # Should predict France
            ('Berlin', 'neighbor_of', '?'),  # No ground truth, should rank neighbors high
        ]

        def predict_tail(model, head, relation, entity2id, relation2id, id2entity, top_k=3):
            """Predict most likely tail entities for (head, relation, ?)."""
            head_id = entity2id[head]
            rel_id = relation2id[relation]

            scores = []
            for tail_entity, tail_id in entity2id.items():
                triple = torch.tensor([[head_id, rel_id, tail_id]], dtype=torch.long)
                score = model(triple).item()
                scores.append((tail_entity, score))

            # Lower score = better in TransE
            scores.sort(key=lambda x: x[1])
            return scores[:top_k]

        id2entity = {i: e for e, i in entity2id.items()}

        print("\nLink Prediction Results:")
        for head, rel, _ in test_queries:
            print(f"\nQuery: ({head}, {rel}, ?)")
            predictions = predict_tail(transe_model, head, rel, entity2id, relation2id, id2entity)
            for rank, (entity, score) in enumerate(predictions, 1):
                print(f"  Rank {rank}: {entity} (score: {score:.3f})")

        # Compute Mean Reciprocal Rank (MRR) on known triples
        def compute_mrr(model, triples, entity2id, relation2id):
            """Mean Reciprocal Rank: average of 1/rank for correct answers."""
            reciprocal_ranks = []

            for h, r, t in triples:
                scores = []
                h_id = entity2id[h]
                r_id = relation2id[r]
                t_id = entity2id[t]

                # Rank all possible tails
                for tail_id in range(len(entity2id)):
                    triple = torch.tensor([[h_id, r_id, tail_id]], dtype=torch.long)
                    score = model(triple).item()
                    scores.append(score)

                # Rank of true tail (lower score = better rank)
                sorted_indices = np.argsort(scores)
                rank = np.where(sorted_indices == t_id)[0][0] + 1
                reciprocal_ranks.append(1.0 / rank)

            return np.mean(reciprocal_ranks)

        mrr = compute_mrr(transe_model, triples[:5], entity2id, relation2id)  # Test on capitals
        print(f"\nMean Reciprocal Rank (MRR): {mrr:.3f}")
        print(f"Average rank: {1/mrr:.1f}")

        print("✓ Part 8 passed")
        return True

    except Exception as e:
        print(f"✗ Part 8 failed: {e}")
        traceback.print_exc()
        return False


def test_part_9():
    """Test Part 9: Graph-Based Recommendation with LightGCN"""
    print("\n" + "="*80)
    print("Testing Part 9: LightGCN Recommendations (reduced epochs)")
    print("="*80)

    try:
        import torch
        import torch.nn.functional as F
        import numpy as np

        torch.manual_seed(42)
        np.random.seed(42)

        # Simplified LightGCN for recommendations
        # Using synthetic user-item interaction data

        # Generate synthetic user-item bipartite graph
        n_users_rec = 100
        n_items = 50
        n_interactions = 500

        # Create interaction matrix
        interactions = []
        for _ in range(n_interactions):
            user = np.random.randint(0, n_users_rec)
            item = np.random.randint(0, n_items)
            interactions.append((user, item))

        # Remove duplicates
        interactions = list(set(interactions))
        n_interactions = len(interactions)

        print(f"Recommendation Dataset:")
        print(f"  Users: {n_users_rec}")
        print(f"  Items: {n_items}")
        print(f"  Interactions: {n_interactions}")
        print(f"  Sparsity: {(1 - n_interactions / (n_users_rec * n_items)) * 100:.1f}%")

        # Build bipartite graph as PyTorch Geometric data
        # Nodes: [user_0, ..., user_99, item_0, ..., item_49]
        # Edges: user-item interactions (bidirectional)

        edge_index_list = []
        for user, item in interactions:
            # User -> Item
            edge_index_list.append([user, n_users_rec + item])
            # Item -> User (bipartite needs both directions)
            edge_index_list.append([n_users_rec + item, user])

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()

        # Split interactions into train/test
        train_size = int(0.8 * n_interactions)
        train_interactions = interactions[:train_size]
        test_interactions = interactions[train_size:]

        # Train edges only
        train_edge_list = []
        for user, item in train_interactions:
            train_edge_list.append([user, n_users_rec + item])
            train_edge_list.append([n_users_rec + item, user])
        train_edge_index = torch.tensor(train_edge_list, dtype=torch.long).t()

        print(f"  Train interactions: {len(train_interactions)}")
        print(f"  Test interactions: {len(test_interactions)}")

        # LightGCN: simplified GCN without feature transformation
        class LightGCN(torch.nn.Module):
            def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3):
                super(LightGCN, self).__init__()
                self.n_users = n_users
                self.n_items = n_items
                self.n_layers = n_layers

                # Learnable embeddings for users and items
                self.user_embedding = torch.nn.Embedding(n_users, embedding_dim)
                self.item_embedding = torch.nn.Embedding(n_items, embedding_dim)

                torch.nn.init.xavier_uniform_(self.user_embedding.weight)
                torch.nn.init.xavier_uniform_(self.item_embedding.weight)

            def forward(self, edge_index):
                # Combine user and item embeddings
                all_embeddings = torch.cat([self.user_embedding.weight,
                                             self.item_embedding.weight], dim=0)

                embeddings_list = [all_embeddings]

                # Layer-wise propagation (no weight transformation, just aggregation)
                for _ in range(self.n_layers):
                    # Normalize by degree
                    row, col = edge_index
                    deg = torch.bincount(row, minlength=all_embeddings.size(0)).float()
                    deg_inv_sqrt = deg.pow(-0.5)
                    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

                    # Aggregation
                    all_embeddings_layer = torch.zeros_like(all_embeddings)
                    for i in range(edge_index.size(1)):
                        src, dst = edge_index[0, i], edge_index[1, i]
                        all_embeddings_layer[dst] += all_embeddings[src] * deg_inv_sqrt[src] * deg_inv_sqrt[dst]

                    embeddings_list.append(all_embeddings_layer)
                    all_embeddings = all_embeddings_layer

                # Average across layers
                final_embeddings = torch.mean(torch.stack(embeddings_list, dim=0), dim=0)

                user_embeddings = final_embeddings[:self.n_users]
                item_embeddings = final_embeddings[self.n_users:]

                return user_embeddings, item_embeddings

            def predict(self, user_ids, item_ids, edge_index):
                user_emb, item_emb = self.forward(edge_index)
                user_emb = user_emb[user_ids]
                item_emb = item_emb[item_ids]
                # Dot product for prediction
                scores = (user_emb * item_emb).sum(dim=1)
                return scores

        lightgcn_model = LightGCN(n_users=n_users_rec, n_items=n_items, embedding_dim=32)
        optimizer = torch.optim.Adam(lightgcn_model.parameters(), lr=0.001)

        # BPR loss: Bayesian Personalized Ranking
        def bpr_loss(pos_scores, neg_scores):
            """Positive items should score higher than negative items."""
            return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        print("\nTraining LightGCN...")
        for epoch in range(1, 51):  # Reduced from 201 to 51
            lightgcn_model.train()
            optimizer.zero_grad()

            # Sample positive and negative pairs
            batch_users = torch.randint(0, n_users_rec, (32,))
            batch_pos_items = torch.randint(0, n_items, (32,))
            batch_neg_items = torch.randint(0, n_items, (32,))

            pos_scores = lightgcn_model.predict(batch_users, batch_pos_items, train_edge_index)
            neg_scores = lightgcn_model.predict(batch_users, batch_neg_items, train_edge_index)

            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()

            if epoch % 25 == 0:
                print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

        print("Training complete.")

        print("✓ Part 9 passed")
        return True, lightgcn_model, test_interactions, train_edge_index, n_items

    except Exception as e:
        print(f"✗ Part 9 failed: {e}")
        traceback.print_exc()
        return False, None, None, None, None


def test_part_10(lightgcn_model=None, test_interactions=None, train_edge_index=None, n_items=None):
    """Test Part 10: Recommendation Evaluation"""
    print("\n" + "="*80)
    print("Testing Part 10: Recommendation Evaluation")
    print("="*80)

    try:
        import torch
        import numpy as np
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import svds

        if lightgcn_model is None:
            # Recreate if not provided
            test_result, lightgcn_model, test_interactions, train_edge_index, n_items = test_part_9()
            if not test_result:
                raise Exception("Failed to create LightGCN model")

        n_users_rec = 100

        # Evaluate recommendations: Recall@K and NDCG@K
        def evaluate_recommendations(model, test_interactions, edge_index, k=10):
            """Compute Recall@K and NDCG@K."""
            model.eval()
            recalls = []
            ndcgs = []

            # Group test interactions by user
            user_items = {}
            for user, item in test_interactions:
                if user not in user_items:
                    user_items[user] = []
                user_items[user].append(item)

            with torch.no_grad():
                user_emb, item_emb = model.forward(edge_index)

                for user, true_items in user_items.items():
                    # Score all items for this user
                    user_vec = user_emb[user].unsqueeze(0)  # [1, dim]
                    scores = (user_vec * item_emb).sum(dim=1)  # [n_items]

                    # Top-K items
                    _, top_k_items = torch.topk(scores, k)
                    top_k_items = top_k_items.cpu().numpy()

                    # Recall@K: fraction of true items in top-K
                    hits = len(set(top_k_items) & set(true_items))
                    recall = hits / len(true_items)
                    recalls.append(recall)

                    # NDCG@K: position-weighted relevance
                    dcg = 0
                    for rank, item in enumerate(top_k_items, 1):
                        if item in true_items:
                            dcg += 1 / np.log2(rank + 1)

                    idcg = sum(1 / np.log2(rank + 1) for rank in range(1, min(len(true_items), k) + 1))
                    ndcg = dcg / idcg if idcg > 0 else 0
                    ndcgs.append(ndcg)

            return np.mean(recalls), np.mean(ndcgs)

        recall_at_10, ndcg_at_10 = evaluate_recommendations(
            lightgcn_model, test_interactions, train_edge_index, k=10
        )

        print(f"\nLightGCN Test Results:")
        print(f"  Recall@10: {recall_at_10:.3f}")
        print(f"  NDCG@10:   {ndcg_at_10:.3f}")

        # Baseline: Matrix Factorization (SVD)
        # Recreate train interactions
        n_interactions_total = 500
        interactions_all = []
        for _ in range(n_interactions_total):
            user = np.random.randint(0, n_users_rec)
            item = np.random.randint(0, n_items)
            interactions_all.append((user, item))
        interactions_all = list(set(interactions_all))
        train_size = int(0.8 * len(interactions_all))
        train_interactions = interactions_all[:train_size]

        # Build interaction matrix
        R = lil_matrix((n_users_rec, n_items))
        for user, item in train_interactions:
            R[user, item] = 1.0
        R = R.tocsr()

        # SVD with 20 latent factors
        U, sigma, Vt = svds(R, k=20)
        sigma_diag = np.diag(sigma)
        R_pred = U @ sigma_diag @ Vt

        def svd_evaluate(R_pred, test_interactions, k=10):
            """Evaluate SVD recommendations."""
            user_items = {}
            for user, item in test_interactions:
                if user not in user_items:
                    user_items[user] = []
                user_items[user].append(item)

            recalls = []
            ndcgs = []

            for user, true_items in user_items.items():
                scores = R_pred[user, :]
                top_k_items = np.argsort(scores)[-k:][::-1]

                hits = len(set(top_k_items) & set(true_items))
                recall = hits / len(true_items)
                recalls.append(recall)

                dcg = sum(1 / np.log2(rank + 2) for rank, item in enumerate(top_k_items) if item in true_items)
                idcg = sum(1 / np.log2(rank + 2) for rank in range(min(len(true_items), k)))
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcgs.append(ndcg)

            return np.mean(recalls), np.mean(ndcgs)

        svd_recall, svd_ndcg = svd_evaluate(R_pred, test_interactions, k=10)

        print(f"\nSVD (Matrix Factorization) Test Results:")
        print(f"  Recall@10: {svd_recall:.3f}")
        print(f"  NDCG@10:   {svd_ndcg:.3f}")

        print(f"\nComparison:")
        print(f"  LightGCN NDCG@10: {ndcg_at_10:.3f}")
        print(f"  SVD NDCG@10:      {svd_ndcg:.3f}")
        improvement = ((ndcg_at_10 - svd_ndcg) / (svd_ndcg + 1e-10) * 100)
        print(f"  Improvement:      {improvement:.1f}%")

        print("✓ Part 10 passed")
        return True

    except Exception as e:
        print(f"✗ Part 10 failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all code tests"""
    print("\n" + "="*80)
    print("CHAPTER 58: APPLICATIONS OF GRAPH ML - CODE REVIEW")
    print("="*80)

    results = {}

    # Part 1: Molecular dataset
    results['Part 1'] = test_part_1()

    # Part 2: MPNN architecture
    results['Part 2'] = test_part_2()

    # Part 3: Training
    test_3_result = test_part_3()
    if isinstance(test_3_result, tuple):
        results['Part 3'] = test_3_result[0]
        _, model, test_rmse, test_preds, test_targets = test_3_result
    else:
        results['Part 3'] = test_3_result
        model, test_rmse, test_preds, test_targets = None, None, None, None

    # Part 4: Baseline comparison
    results['Part 4'] = test_part_4(test_rmse, test_preds, test_targets)

    # Part 5: Fraud detection dataset
    test_5_result = test_part_5()
    if isinstance(test_5_result, tuple):
        results['Part 5'] = test_5_result[0]
        _, fraud_data = test_5_result
    else:
        results['Part 5'] = test_5_result
        fraud_data = None

    # Part 6: Fraud detection model
    results['Part 6'] = test_part_6(fraud_data)

    # Part 7: TransE
    test_7_result = test_part_7()
    if isinstance(test_7_result, tuple):
        results['Part 7'] = test_7_result[0]
        _, transe_model, entity2id, relation2id, triples = test_7_result
    else:
        results['Part 7'] = test_7_result
        transe_model, entity2id, relation2id, triples = None, None, None, None

    # Part 8: Link prediction
    results['Part 8'] = test_part_8(transe_model, entity2id, relation2id, triples)

    # Part 9: LightGCN
    test_9_result = test_part_9()
    if isinstance(test_9_result, tuple):
        results['Part 9'] = test_9_result[0]
        _, lightgcn_model, test_interactions, train_edge_index, n_items = test_9_result
    else:
        results['Part 9'] = test_9_result
        lightgcn_model, test_interactions, train_edge_index, n_items = None, None, None, None

    # Part 10: Recommendation evaluation
    results['Part 10'] = test_part_10(lightgcn_model, test_interactions, train_edge_index, n_items)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for part, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{part}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        return 0
    else:
        print(f"\n✗✗✗ {total - passed} TEST(S) FAILED ✗✗✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
