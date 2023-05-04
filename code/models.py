import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphConvolution

class GCN(nn.Module):
    """
    A Graph Convolutional Network (GCN) module that applies two graph convolutional layers to a sparse adjacency matrix.
    """
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        """
        Initializes a Graph Convolutional Network (GCN) model with two graph convolutional layers and dropout.

        Args:
            voc_size (int): The size of the vocabulary.
            emb_dim (int): The dimensionality of the embedding space.
            adj (np.ndarray): An adjacency matrix representing the graph structure of the data.
            device (torch.device): The device on which to perform computations. Default is 'cpu:0'.

        Returns:
            None.
        """
        super().__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        # Normalize the adjacency matrix
        adj = self.normalize(adj + np.eye(adj.shape[0]))

        # Initialize the adjacency and identity matrices as model parameters
        self.adj = nn.Parameter(torch.FloatTensor(adj).to(device), requires_grad=False)
        self.x = nn.Parameter(torch.eye(voc_size).to(device), requires_grad=False)

        # Define the graph convolutional layers and the dropout layer
        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        """
        Perform a forward pass through the graph convolutional layers.

        Returns:
            torch.FloatTensor: The node embeddings after the second graph convolutional layer.
        """
        # Perform the first graph convolutional layer and apply ReLU activation function
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)

        # Apply dropout regularization
        node_embedding = self.dropout(node_embedding)

        # Perform the second graph convolutional layer
        node_embedding = self.gcn2(node_embedding, self.adj)

        return node_embedding

    def normalize(self, mx):
        """
        Row-normalize a sparse matrix.

        Args:
            mx (ndarray): The sparse matrix to be normalized.

        Returns:
            mx (ndarray): The row-normalized sparse matrix.
        """
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class GAMENet(nn.Module):
    """
    GAMENet model for patient representation learning and drug-drug interaction prediction.
    """
    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0'), ddi_in_memory=True):
        """
        Initializes the GAMENet model.

        Args:
            vocab_size (list): A list of integers representing the size of the vocabulary for each input data type. The first two values in the list correspond to the size of the vocabulary for EHR and DDI data, respectively, while the third value corresponds to the size of the vocabulary for the output data (ADRs).
            ehr_adj (numpy.ndarray): An adjacency matrix for the EHR data.
            ddi_adj (numpy.ndarray): An adjacency matrix for the DDI data.
            emb_dim (int): The dimension of the embedding vectors.
            device (torch.device): The device on which to run the model.
            ddi_in_memory (bool): Whether to store the DDI adjacency matrix in memory. 

        Returns:
            None.
        """
        super().__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory

        # Embedding layer
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K-1)])
        self.dropout = nn.Dropout(p=0.4)

        # Encoder layer
        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim*2, batch_first=True) for _ in range(K-1)])

        # Query layer
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        # GCN layers for EHR and DDI
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        # Output layer
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

        self.init_weights()

    def forward(self, input):
        """
        Forward pass of the GAMENet model.

        Args:
            input (list): A list of tuples representing patient admissions. Each tuple contains three lists: 
                - drug codes for the current admission, 
                - drug codes for previous admissions, 
                - codes of drug-drug interactions in the current admission.

        Returns:
            output (torch.Tensor): Output tensor of shape (1, dim).
            batch_neg (torch.Tensor, optional): Negative batch prediction probability tensor of shape (voc_size, voc_size).

        """
        # Generate medical embeddings and queries
        i1_seq = []
        i2_seq = []

        def mean_embedding(embedding):
            """
            Computes the mean of the input tensor along dim=1 and adds a new dimension with size 1 at dim=0.

            Args:
                embedding (torch.Tensor): Tensor of shape (batch_size, seq_len, dim).

            Returns:
                torch.Tensor: Tensor of shape (1, 1, dim).
            """
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        
        for adm in input:
            i1 = mean_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            i2 = mean_embedding(self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i1_seq.append(i1)
            i2_seq.append(i2)

        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)

        o1, h1 = self.encoders[0](
            i1_seq
        ) # o1:(1, seq, dim*2) hi:(1,1,dim*2)
        o2, h2 = self.encoders[1](
            i2_seq
        )
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0) # (seq, dim*4)
        queries = self.query(patient_representations) # (seq, dim)

        # Graph Memory Module
        '''I:generate current input'''
        query = queries[-1:] # (1,dim)

        '''G:generate graph memory bank and insert history information'''
        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)
        else:
            drug_memory = self.ehr_gcn()

        if len(input) > 1:
            history_keys = queries[:(queries.size(0)-1)] # (seq-1, dim)

            history_values = np.zeros((len(input)-1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device) # (seq-1, size)

        '''O:read from global memory bank and dynamic memory bank'''
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t())) # (1, seq-1)
            weighted_values = visit_weight.mm(history_values) # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
        else:
            fact2 = fact1
        
        '''R:convert O and predict'''
        output = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)

        if self.training:
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            return output

    def init_weights(self):
        """
        Initialize weights of the embeddings and inter parameter.
        """
        # Set the range for the initial uniform distribution
        initrange = 0.1
        
        # Initialize the weights of the embeddings
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        
        # Initialize the inter parameter
        self.inter.data.uniform_(-initrange, initrange)


'''
Leap
'''
class Leap(nn.Module):
    """
    The Leap class represents a Language Embedding Attention Pointer (LEAP) model. 
    """
    def __init__(self, voc_size, emb_dim=128, device=torch.device('cpu:0')):
        """
        Initialize a Language Embedding Attention Pointer (LEAP) model.
        
        Args:
            voc_size (tuple): a tuple containing the sizes of the vocabularies for the source 
                            and target languages, and the size of the joint vocabulary.
            emb_dim (int): the size of the embedding vectors.
            device (torch.device): the device on which to perform computations.

        Returns:
            None.
        """
        super(Leap, self).__init__()

        self.voc_size = voc_size
        self.device = device
        self.SOS_TOKEN = voc_size[2]
        self.END_TOKEN = voc_size[2]+1

        self.enc_embedding = nn.Sequential(
            nn.Embedding(voc_size[0], emb_dim, ),
            nn.Dropout(0.3)
        )

        self.dec_embedding = nn.Sequential(
            nn.Embedding(voc_size[2] + 2, emb_dim, ),
            nn.Dropout(0.3)
        )

        self.dec_gru = nn.GRU(emb_dim*2, emb_dim, batch_first=True)

        self.attn = nn.Linear(emb_dim*2, 1)

        self.output = nn.Linear(emb_dim, voc_size[2]+2)


    def forward(self, input, max_len=20):
        """
        Forward pass of the LEAP model.

        Args:
            input: A tuple of size 3, where the first element is a list of medical codes, the second element is the length
                 of the codes list, and the third element is the initial state for the decoder.
            max_len: Maximum length of the output sequence.

        Returns:
            A tensor of shape (max_len, voc_size[2]+2) representing the logits of the output sequence.
        """
        device = self.device
        # input (3, codes)
        input_tensor = torch.LongTensor(input[0]).to(device)
        # (len, dim)
        input_embedding = self.enc_embedding(input_tensor.unsqueeze(dim=0)).squeeze(dim=0)

        output_logits = []
        hidden_state = None
        if self.training:
            # Teacher forcing mode
            for med_code in [self.SOS_TOKEN] + input[2]:
                dec_input = torch.LongTensor([med_code]).unsqueeze(dim=0).to(device)
                dec_input = self.dec_embedding(dec_input).squeeze(dim=0) # (1,dim)

                if hidden_state is None:
                    hidden_state = dec_input

                hidden_state_repeat = hidden_state.repeat(input_embedding.size(0), 1) # (len, dim)
                combined_input = torch.cat([hidden_state_repeat, input_embedding], dim=-1) # (len, dim*2)
                attn_weight = F.softmax(self.attn(combined_input).t(), dim=-1) # (1, len)
                input_embedding = attn_weight.mm(input_embedding) # (1, dim)

                _, hidden_state = self.dec_gru(torch.cat([input_embedding, dec_input], dim=-1).unsqueeze(dim=0), hidden_state.unsqueeze(dim=0))
                hidden_state = hidden_state.squeeze(dim=0) # (1,dim)

                output_logits.append(self.output(F.relu(hidden_state)))

            return torch.cat(output_logits, dim=0)

        else:
            # Inference mode
            for di in range(max_len):
                if di == 0:
                    dec_input = torch.LongTensor([[self.SOS_TOKEN]]).to(device)
                dec_input = self.dec_embedding(dec_input).squeeze(dim=0) # (1,dim)
                if hidden_state is None:
                    hidden_state = dec_input
                hidden_state_repeat = hidden_state.repeat(input_embedding.size(0), 1)  # (len, dim)
                combined_input = torch.cat([hidden_state_repeat, input_embedding], dim=-1)  # (len, dim*2)
                attn_weight = F.softmax(self.attn(combined_input).t(), dim=-1)  # (1, len)
                input_embedding = attn_weight.mm(input_embedding)  # (1, dim)
                _, hidden_state = self.dec_gru(torch.cat([input_embedding, dec_input], dim=-1).unsqueeze(dim=0),
                                               hidden_state.unsqueeze(dim=0))
                hidden_state = hidden_state.squeeze(dim=0)  # (1,dim)
                output = self.output(F.relu(hidden_state))
                topv, topi = output.data.topk(1)
                output_logits.append(F.softmax(output, dim=-1))
                dec_input = topi.detach()
            return torch.cat(output_logits, dim=0)

'''
Retain
'''
class Retain(nn.Module):
    """
    RETAIN (REverse Time AttentIoN) is a type of neural network model used in healthcare and clinical applications for predicting patient outcomes. 
    """
    def __init__(self, voc_size, emb_size=64, device=torch.device('cpu:0')):
        """
        Initialize the Retain model.

        Args:
            voc_size (tuple): A tuple containing the number of unique codes for each code category.
            emb_size (int): The dimensionality of the code embeddings.
            device (torch.device): The device to be used for computations.

        Returns:
            None.
        """
        super(Retain, self).__init__()
        self.device = device
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.input_len = voc_size[0] + voc_size[1] + voc_size[2]
        self.output_len = voc_size[2]

        self.embedding = nn.Sequential(
            nn.Embedding(self.input_len + 1, self.emb_size, padding_idx=self.input_len),
            nn.Dropout(0.3)
        )

        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru = nn.GRU(emb_size, emb_size, batch_first=True)

        self.alpha_li = nn.Linear(emb_size, 1)
        self.beta_li = nn.Linear(emb_size, emb_size)

        self.output = nn.Linear(emb_size, self.output_len)

    def forward(self, input):
        """
        Compute the forward pass of the Retain model.

        Args:
            input (list): A list of patient visit data, where each visit is represented as a tuple
                containing three lists of codes (one for each code category).
        
        Returns:
            output (torch.Tensor): The output logits of the Retain model, representing the predicted
                probabilities of each code in the third code category for each patient in the input.
        """
        device = self.device
        # input: (visit, 3, codes )
        max_len = max([(len(v[0]) + len(v[1]) + len(v[2])) for v in input])
        input_np = []
        for visit in input:
            input_tmp = []
            input_tmp.extend(visit[0])
            input_tmp.extend(list(np.array(visit[1]) + self.voc_size[0]))
            input_tmp.extend(list(np.array(visit[2]) + self.voc_size[0] + self.voc_size[1]))
            if len(input_tmp) < max_len:
                input_tmp.extend( [self.input_len]*(max_len - len(input_tmp)) )

            input_np.append(input_tmp)

        visit_emb = self.embedding(torch.LongTensor(input_np).to(device)) # (visit, max_len, emb)
        visit_emb = torch.sum(visit_emb, dim=1) # (visit, emb)

        g, _ = self.alpha_gru(visit_emb.unsqueeze(dim=0)) # g: (1, visit, emb)
        h, _ = self.beta_gru(visit_emb.unsqueeze(dim=0)) # h: (1, visit, emb)

        g = g.squeeze(dim=0) # (visit, emb)
        h = h.squeeze(dim=0) # (visit, emb)
        attn_g = F.softmax(self.alpha_li(g), dim=-1) # (visit, 1)
        attn_h = F.tanh(self.beta_li(h)) # (visit, emb)

        c = attn_g * attn_h * visit_emb # (visit, emb)
        c = torch.sum(c, dim=0).unsqueeze(dim=0) # (1, emb)

        return self.output(c)
