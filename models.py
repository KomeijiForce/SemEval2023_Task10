import torch
from torch import nn
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam, AdamW
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.metrics import f1_score

class SexismDetector(nn.Module):
    
    def __init__(self, config):
        
        super().__init__()
        
        path = config['path']

        self.labels = config["labels"]
        
        self.device = torch.device(config['device'])
        
        self.loss_weight = config['loss_weight']
        
        self.disable_neg_train = config['disable_neg_train']
        self.disable_neg_eval = config['disable_neg_eval']
        
        self.batch_size = config['batch_size']
        
        self.tok = AutoTokenizer.from_pretrained(path)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(path, num_labels=len(self.labels["label_vector"]))

        self.classifier = self.classifier.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam([p for p in self.classifier.parameters()], lr=config['lr'], eps=1e-6, betas=(0.9, 0.999))
        self.scheduler = ExponentialLR(self.optimizer, .67**(1/5000))
        
        self.build_transfer_matrix()
    
    def forward(self, df_batch):

        texts = list(df_batch.text.values)

        inputs = self.tok(texts, padding=True, return_tensors='pt')
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)
        scores_vector = self.classifier(**inputs)[-1]

        return scores_vector

    def train(self, df, config):

        bar = tqdm(range(0, len(df), self.batch_size))

        for idx in bar:
            df_batch = df[idx:idx+self.batch_size]
            scores_vector = self(df_batch)

            if self.disable_neg_train:
                scores_vector[0] = -1e8

            q = {}

            q['vector'] = scores_vector.softmax(-1)
            q['category'] = torch.mm(q['vector'], self.transfer_category)
            q['sexist'] = torch.mm(q['vector'], self.transfer_sexist)
            y = self.build_y(df_batch)

            self.classifier.zero_grad()

            loss = sum([self.criterion(q[key].log(), y[f"label_{key}"]) * self.loss_weight[key] for key in q.keys()])

            if config["gradient_accumulation_steps"] > 1:
                loss = loss / config["gradient_accumulation_steps"]

            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            bar.set_description(f'@Train #Loss={loss:.4}')

        print(f'@Train #Loss={loss:.4}')            

    def evaluate(self, df, target, dataset="Dev"):

        bar = tqdm(range(0, len(df), self.batch_size))

        P, Y = torch.LongTensor([]).to(self.device), torch.LongTensor([]).to(self.device)

        with torch.no_grad():
            for idx in bar:
                df_batch = df[idx:idx+self.batch_size]
                scores_vector = self(df_batch)

                if self.disable_neg_eval:
                    scores_vector[0] = -1e8

                q = {}
                q['vector'] = scores_vector.softmax(-1)
                q['category'] = torch.mm(q['vector'], self.transfer_category)
                q['sexist'] = torch.mm(q['vector'], self.transfer_sexist)

                p = {key:q[key].argmax(-1) for key in q.keys()}

                y = self.build_y(df_batch)

                P = torch.cat([P, p[target]], 0)
                Y = torch.cat([Y, y[f"label_{target}"]], 0)

                s = self.get_score(P, Y)

                bar.set_description(f'@Evaluate {dataset} #{target.capitalize()} F1={s:.4}')
            print(f'\t\tEvaluate {dataset} #{target.capitalize()} F1={s:.4}')

        return s

    def get_score(self, y, p):
        return f1_score(y.detach().cpu().numpy(), p.detach().cpu().numpy(), average='macro')
    
    def build_transfer_matrix(self):
        self.transfer_category = torch.zeros((len(self.labels["label_vector"]), len(self.labels["label_category"]))).to(self.device)
        self.transfer_category[0, 0] = 1
        self.transfer_category[1:3, 1] = 1
        self.transfer_category[3:6, 2] = 1
        self.transfer_category[6:10, 3] = 1
        self.transfer_category[10:12, 4] = 1

        self.transfer_sexist = torch.zeros((len(self.labels["label_vector"]), len(self.labels["label_sexist"]))).to(self.device)
        self.transfer_sexist[0, 0] = 1
        self.transfer_sexist[1:, 1] = 1
        
    def build_y(self, df_batch):
    
        y = {
            "label_sexist":[],
            "label_category":[],
            "label_vector":[],
        }

        for label_vector in df_batch.label_vector.values:
            if label_vector == "none":
                keys = ["0", "0", "0"]
            else:
                keys = ["1", label_vector.split(".")[0], label_vector.split(" ")[0]]

            for key, lkey in zip(keys, y.keys()):
                y[lkey].append(list(self.labels[lkey].values()).index(self.labels[lkey][key]))

        for lkey in y.keys():
            y[lkey] = torch.LongTensor(y[lkey]).to(self.device)

        return y
