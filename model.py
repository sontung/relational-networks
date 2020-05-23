import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvInputModel(nn.Module):
    def __init__(self, nb_input_channels=512):
        super(ConvInputModel, self).__init__()

        self.conv1 = nn.Conv2d(nb_input_channels, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)

        if nb_input_channels == 3:
            self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
            self.batchNorm2 = nn.BatchNorm2d(24)
            self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
            self.batchNorm3 = nn.BatchNorm2d(24)
            self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
            self.batchNorm4 = nn.BatchNorm2d(24)

    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)

        if img.size(2) > 16:
            x = self.conv2(x)
            x = F.relu(x)
            x = self.batchNorm2(x)

            x = self.conv3(x)
            x = F.relu(x)
            x = self.batchNorm3(x)

            x = self.conv4(x)
            x = F.relu(x)
            x = self.batchNorm4(x)
        return x


class RNforPixel(nn.Module):
    def __init__(self, vocab_size=83, word_embedding_dim=32, lstm_hidden_size=128, nb_input_channels=512,
                 feature_size_after_conv=24, nb_classes=28):
        super(RNforPixel, self).__init__()

        self.cuda = True
        self.conv = ConvInputModel(nb_input_channels)

        self.text_inp = nn.Embedding(vocab_size, word_embedding_dim)
        self.lstm = nn.LSTM(word_embedding_dim, lstm_hidden_size, batch_first=True)

        self.relation_type = "bin"

        self.g_out = nn.Sequential(
            nn.Linear((feature_size_after_conv + 2) * 2 + lstm_hidden_size, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
        )

        self.fcout = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, nb_classes)
        )

    def forward(self, img, qu, dev="cuda"):
        x = self.conv(img)  # [64, 24, 8, 8]

        batch_size = x.size(0)
        n_channels = x.size(1)
        d = x.size(2)  # how many entities per x or per y
        x_flat = x.view(batch_size, n_channels, d * d)  # [64, 24, 64]
        nb_entities = d**2
        coord_tensor = self.prepare_coord_tensor2(batch_size, d).view(batch_size, 2, nb_entities)
        coord_tensor = coord_tensor.to(dev)

        # add coordinates
        x_flat = torch.cat([x_flat, coord_tensor], 1)  # [64, 24+2, 64]
        x_flat = x_flat.permute(0, 2, 1)  # [64, 64, 24+2]

        # process question
        qu = self.text_inp(qu)
        _, (feature_vector_txt, _) = self.lstm(qu)
        qst = feature_vector_txt.squeeze()

        # add question everywhere
        qst = torch.unsqueeze(qst, 1)
        qst = qst.repeat(1, nb_entities, 1)
        qst = torch.unsqueeze(qst, 2)  # [64, 64, 1, lstm hidden=128]

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)
        x_i = x_i.repeat(1, nb_entities, 1, 1)
        x_j = torch.unsqueeze(x_flat, 2)
        x_j = torch.cat([x_j, qst], 3)
        x_j = x_j.repeat(1, 1, nb_entities, 1)

        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3)  # [64, 64, 64, 180]
        total_information = x_full.size(3)

        # reshape for passing through network
        x_ = x_full.view(batch_size * (d * d) * (d * d), total_information)  # [-1, lstm hidden+26*2]
        x_ = self.g_out(x_)  # [262144, 256]

        # reshape again and sum
        x_g = x_.view(batch_size, (d * d) * (d * d), 256)
        x_g = x_g.sum(1).squeeze()

        return self.fcout(x_g)

    @staticmethod
    def prepare_coord_tensor2(b, d):
        coords = torch.linspace(-d / 2., d / 2., d)
        x = coords.unsqueeze(0).repeat(d, 1)
        y = coords.unsqueeze(1).repeat(1, d)
        ct = torch.stack((x, y))
        ct = ct.unsqueeze(0).repeat(b, 1, 1, 1)
        return ct.clone().detach()

    def inference(self, im_, questions_, dev):
        self.eval()
        with torch.no_grad():
            logits = self.forward(im_, questions_, dev)
        return torch.argmax(logits, 1)


class RNsmall(nn.Module):
    def __init__(self, vocab_size=83, word_embedding_dim=32, lstm_hidden_size=128,
                 feature_size_after_conv=24, nb_classes=28):
        super(RNsmall, self).__init__()
        self.cuda = True

        self.text_inp = nn.Embedding(vocab_size, word_embedding_dim)
        self.lstm = nn.LSTM(word_embedding_dim, lstm_hidden_size, batch_first=True)

        self.relation_type = "bin"

        self.g_out = nn.Sequential(
            nn.Linear((feature_size_after_conv + 2) * 2 + lstm_hidden_size, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
        )

        self.fcout = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, nb_classes)
        )

    def load_pretrained(self, dir_=None):
        if dir_ is None:
            dir_ = "/home/sontung/thesis/RelationNetworks-CLEVR/pretrained_models/original_fp_epoch_493.pth"
        pre_trained = torch.load(dir_, map_location="cpu")

        new_state_dict = {
            "text_inp.weight": pre_trained["module.text.wembedding.weight"],
            "lstm.weight_ih_l0": pre_trained["module.text.lstm.weight_ih_l0"],
            "lstm.weight_hh_l0": pre_trained["module.text.lstm.weight_hh_l0"],
            "lstm.bias_ih_l0": pre_trained["module.text.lstm.bias_ih_l0"],
            "lstm.bias_hh_l0": pre_trained["module.text.lstm.bias_hh_l0"],
            # "g_out.0.weight": pre_trained["module.rl.g_layers.0.weight"],
            # "g_out.0.bias": pre_trained["module.rl.g_layers.0.bias"],
            # "g_out.1.weight": pre_trained["module.rl.g_layers.1.weight"],
            # "g_out.1.bias": pre_trained["module.rl.g_layers.1.bias"],
            # "g_out.2.weight": pre_trained["module.rl.g_layers.2.weight"],
            # "g_out.2.bias": pre_trained["module.rl.g_layers.2.bias"],
            # "g_out.3.weight": pre_trained["module.rl.g_layers.3.weight"],
            # "g_out.3.bias": pre_trained["module.rl.g_layers.3.bias"],
            # "fcout.0.weight": pre_trained["module.rl.f_fc1.weight"],
            # "fcout.0.bias": pre_trained["module.rl.f_fc1.bias"],
            # "fcout.2.weight": pre_trained["module.rl.f_fc2.weight"],
            # "fcout.2.bias": pre_trained["module.rl.f_fc2.bias"],
            # "fcout.5.weight": pre_trained["module.rl.f_fc3.weight"],
            # "fcout.5.bias": pre_trained["module.rl.f_fc3.bias"],
        }

        unloaded = ["g_out.0.weight", "g_out.0.bias", "g_out.1.weight", "g_out.1.bias",
                    "g_out.2.weight", "g_out.2.bias", "g_out.3.weight", "g_out.3.bias",
                    "fcout.0.weight", "fcout.0.bias", "fcout.2.weight", "fcout.2.bias",
                    "fcout.5.weight", "fcout.5.bias"]
        for _du in unloaded:
            new_state_dict[_du] = self.state_dict()[_du]
        self.load_state_dict(new_state_dict)
        for __param in self.text_inp.parameters():
            __param.requires_grad = False
        for __param in self.lstm.parameters():
            __param.requires_grad = False

    def forward(self, img, qu):
        x = img  # [64, 24, 8, 8]

        batch_size = x.size(0)
        n_channels = x.size(1)
        d = x.size(2)  # how many entities per x or per y
        x_flat = x.view(batch_size, n_channels, d * d)  # [64, 24, 64]
        nb_entities = d ** 2
        coord_tensor = self.prepare_coord_tensor2(batch_size, d).view(batch_size, 2, nb_entities)
        coord_tensor = coord_tensor.to(img.device)

        # add coordinates
        x_flat = torch.cat([x_flat, coord_tensor], 1)  # [64, 24+2, 64]
        x_flat = x_flat.permute(0, 2, 1)  # [64, 64, 24+2]

        # process question
        qu = self.text_inp(qu)
        _, (feature_vector_txt, _) = self.lstm(qu)
        qst = feature_vector_txt.squeeze()

        # add question everywhere
        qst = torch.unsqueeze(qst, 1)
        qst = qst.repeat(1, nb_entities, 1)
        qst = torch.unsqueeze(qst, 2)  # [64, 64, 1, lstm hidden=128]

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)
        x_i = x_i.repeat(1, nb_entities, 1, 1)
        x_j = torch.unsqueeze(x_flat, 2)
        x_j = torch.cat([x_j, qst], 3)
        x_j = x_j.repeat(1, 1, nb_entities, 1)

        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3)  # [64, 64, 64, 180]
        total_information = x_full.size(3)

        # reshape for passing through network
        x_ = x_full.view(batch_size * (d * d) * (d * d), total_information)  # [-1, lstm hidden+26*2]
        x_ = self.g_out(x_)  # [262144, 256]

        # reshape again and sum
        x_g = x_.view(batch_size, (d * d) * (d * d), 256)
        x_g = x_g.sum(1).squeeze()

        return self.fcout(x_g)

    @staticmethod
    def prepare_coord_tensor2(b, d):
        coords = torch.linspace(-d / 2., d / 2., d)
        x = coords.unsqueeze(0).repeat(d, 1)
        y = coords.unsqueeze(1).repeat(1, d)
        ct = torch.stack((x, y))
        ct = ct.unsqueeze(0).repeat(b, 1, 1, 1)
        return ct.clone().detach()


if __name__ == "__main__":
    # model2 = RNforPixel()
    # model2.to("cuda")
    # out2 = model2(torch.randn((64, 512, 16, 16)).cuda(), torch.ones((64, 49)).cuda().long())
    # print(out2.shape)
    model = RNsmall()
