from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import argparse
import torch
from dataloader import *
from utils import *
from Transformer import TransformerPredictor


def _str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='generte covid data')

parser.add_argument("--task", type=str, default="haniffa")
parser.add_argument("--relabeled", type=_str2bool, default=True)
parser.add_argument('--pca', type=_str2bool, default=True)

parser.add_argument("--test_sample_cells", type=int, default=500,
                    help="number of cells in one sample in test dataset")

parser.add_argument("--test_num_sample", type=int, default=100,
                    help="number of sampled data points in test dataset")

# TransformerPredictor args
parser.add_argument('--emb_dim', type=int, default=128)  # embedding dim
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.3)  # dropout
parser.add_argument('--norm_first', type=_str2bool, default=False)

args = parser.parse_args()

p_idx, labels_, cell_type, patient_id, data, cell_type_large = Covid_data(args)

# print("Orig Labels")
# print(labels_)
# print(labels_.shape)
individual_test = sampling_test_all_only(args, p_idx, labels_)
# print(individual_test)

x_test = []
y_test = []
id_test = []

print("starting")

temp_idx = np.arange(len(p_idx))
# print(temp_idx)

for t_idx in temp_idx:
    id, label = [id_l[0] for id_l in individual_test[t_idx]], [id_l[1] for id_l in individual_test[t_idx]]
    x_test.append([ii for ii in id])
    y_test.append(label[0])
    id_test.append(id)

y_test = np.array(y_test).reshape([-1, 1])
# print("Label")
# print(label)

print("init TransformerPredictor")
best_model = TransformerPredictor(input_dim=50, model_dim=args.emb_dim, num_classes=1,
                              num_heads=args.heads, num_layers=args.layers, dropout=args.dropout,
                              input_dropout=0, pca=args.pca, norm_first=args.norm_first)
# Evaluation

print("loading pre-trained weights")
state_dict = torch.load("artifacts/best_model_weights.pth")
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

best_model.load_state_dict(new_state_dict)
best_model.eval()
pred = []
test_id = []
true = []
wrong = []
prob = []

# print("Loading Test Dataset")
# print("y_test type and shape")
# print(type(y_test))
# print(type(y_test[0]))
# print(y_test[0].shape)

dataset_2 = MyDataset(None, None, x_test, None, None, y_test, None, id_test, fold='test')
test_loader = torch.utils.data.DataLoader(dataset_2, batch_size=1, shuffle=False, collate_fn=dataset_2.collate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigmoid = torch.nn.Sigmoid().to(device)

wrongs = []
label_dict = {0: 'Mild', 1: 'Severe'}
max_acc, max_epoch, max_auc, max_loss, max_valid_acc, max_valid_auc = 0, 0, 0, 0, 0, 0
test_accs, valid_aucs, train_losses, valid_losses, train_accs, test_aucs = [], [0.], [], [], [], []

print("Running Inference...")
with torch.no_grad():
    for batch in (test_loader):
        x_ = torch.from_numpy(data[batch[0]]).float().to(device).squeeze(0)
        y_ = batch[1].int().numpy()
        id_ = batch[2][0]

        out = best_model(x_)
        out = sigmoid(out)
        out = out.detach().cpu().numpy().reshape(-1)

        y_ = y_[0][0]
        true.append(y_)

        prob.append(out[0])

        # majority voting
        f = lambda x: 1 if x > 0.5 else 0
        func = np.vectorize(f)
        out = np.argmax(np.bincount(func(out).reshape(-1))).reshape(-1)[0]
        pred.append(out)
        test_id.append(patient_id[batch[2][0][0][0]])
        if out != y_:
            wrong.append(patient_id[batch[2][0][0][0]])

if len(wrongs) == 0:
    wrongs = set(wrong)
else:
    wrongs = wrongs.intersection(set(wrong))

test_auc = metrics.roc_auc_score(true, prob)

test_acc = accuracy_score(true, pred)
for idx in range(len(pred)):
    print(f"{test_id[idx]} -- true: {label_dict[true[idx]]} -- pred: {label_dict[pred[idx]]}")
test_accs.append(test_acc)

cm = confusion_matrix(true, pred).ravel()
recall = cm[3] / (cm[3] + cm[2])
precision = cm[3] / (cm[3] + cm[1])
if (cm[3] + cm[1]) == 0:
    precision = 0

print("Best performance: Epoch %d, Loss %f, Test ACC %f, Test AUC %f, Test Recall %f, Test Precision %f" % (
max_epoch, max_loss, test_acc, test_auc, recall, precision))
print("Confusion Matrix: " + str(cm))

# Evaluation END