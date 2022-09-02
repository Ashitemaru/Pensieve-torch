import torch
import torch.nn.functional as F

from config import config

def create_scalar_conv1d(out_channels, kernel_size):
    # Create net
    conv = torch.nn.Conv1d(
        in_channels = 1,
        out_channels = out_channels,
        kernel_size = kernel_size,
    )
    
    # Initialize param
    torch.nn.init.xavier_uniform_(conv.weight.data)
    torch.nn.init.constant_(conv.bias.data, 0.0)
    return conv

def create_fc(in_features, out_features):
    # Create net
    fc = torch.nn.Linear(
        in_features = in_features,
        out_features = out_features,
    )

    # Initialize param
    torch.nn.init.xavier_uniform_(fc.weight.data)
    torch.nn.init.constant_(fc.bias.data, 0.0)
    return fc

class ActorNet(torch.nn.Module):
    def __init__(
        self,
        n_feature,
        n_action,
        n_conv_filter = config["n_conv_filter"],
        kernel_size = config["conv_kernel_size"],
        first_fc_output_dim = config["first_fc_output_dim"],
        hidden_fc_output_dim = config["hidden_fc_output_dim"],
    ):
        """
        n_feature.size(): (param_count, past_video_chunk_num)
        param_count: 6.
        past_video_chunk_num: default to 8, denoted as 'k' according to the paper.

        Note:
        Conv1d(in_channels, out_channels, kernel_size)
        Input size: (batch_size, in_channels, embed_size)
        Output size: (batch_size, out_channels, embed_size - kernel_size + 1)
        """

        super(ActorNet, self).__init__()
        self.n_feature = n_feature
        self.n_action = n_action
        self.n_conv_filter = n_conv_filter
        self.first_fc_output_dim = first_fc_output_dim
        self.hidden_fc_output_dim = hidden_fc_output_dim
        self.kernel_size = kernel_size

        # First layers
        self.throughput_conv = create_scalar_conv1d(self.n_conv_filter, self.kernel_size) # Past chunk throughput
        self.time_conv = create_scalar_conv1d(self.n_conv_filter, self.kernel_size) # Past chunk download time
        self.chunk_size_conv = create_scalar_conv1d(self.n_conv_filter, self.kernel_size) # Next chunk sizes
        self.buffer_size_fc = create_fc(1, self.first_fc_output_dim) # Current buffer size
        self.chunk_left_fc = create_fc(1, self.first_fc_output_dim) # Number of chunks left
        self.bitrate_fc = create_fc(1, self.first_fc_output_dim) # Last chunk bitrate

        hidden_layer_input_dim = \
            3 * self.first_fc_output_dim + \
            2 * self.n_conv_filter * (self.n_feature[1] - self.kernel_size + 1) + \
            self.n_conv_filter * (self.n_action - self.kernel_size + 1)

        # Hidden & output layers
        self.hidden_fc = create_fc(hidden_layer_input_dim, self.hidden_fc_output_dim)
        self.output_fc = create_fc(self.hidden_fc_output_dim, self.n_action)

    def forward(self, x):
        """
        x.size(): (batch_size, param_count, past_video_chunk_num)
        batch_size: default to 1.
        param_count: 6, the order of these params are:
            - last chunk bitrate (l)
            - current buffer size (b)
            - past chunk throughput (x)
            - past chunk download time (t)
            - next chunk sizes (n)
            - number of chunks left (c)
        past_video_chunk_num: default to 8, denoted as 'k' according to the paper.
        """

        # First layer
        bitrate_fc_out = F.relu(self.bitrate_fc(x[:, 0: 1, -1]), inplace = True)
        buffer_size_fc_out = F.relu(self.buffer_size_fc(x[:, 1: 2, -1]), inplace = True)
        throughput_conv_out = F.relu(self.throughput_conv(x[:, 2: 3, :]), inplace = True)
        time_conv_out = F.relu(self.time_conv(x[:, 3: 4, :]), inplace = True)
        chunk_size_out = F.relu(self.chunk_size_conv(x[:, 4: 5, : self.n_action]), inplace = True)
        chunk_left_out = F.relu(self.chunk_left_fc(x[:, 5: 6, -1]), inplace = True)

        # Flatten to feed into next layer
        throughput_conv_out_flatten = throughput_conv_out.view(throughput_conv_out.shape[0], -1)
        time_conv_out_flatten = time_conv_out.view(time_conv_out.shape[0], -1)
        chunk_size_out_flatten = chunk_size_out.view(chunk_size_out.shape[0], -1)

        # Concat
        hidden_fc_in = torch.cat([
            bitrate_fc_out,
            buffer_size_fc_out,
            throughput_conv_out_flatten,
            time_conv_out_flatten,
            chunk_size_out_flatten,
            chunk_left_out,
        ], dim = 1)

        # Hidden & output layer
        hidden_fc_out = F.relu(self.hidden_fc(hidden_fc_in), inplace = True)
        return torch.softmax(self.output_fc(hidden_fc_out), dim = -1) # Shape: (batch_size, n_action)

class CriticNet(torch.nn.Module):
    def __init__(
        self,
        n_feature,
        n_action,
        n_conv_filter = config["n_conv_filter"],
        kernel_size = config["conv_kernel_size"],
        first_fc_output_dim = config["first_fc_output_dim"],
        hidden_fc_output_dim = config["hidden_fc_output_dim"],
    ):
        """
        Critic network has the similar structure as the actor network.
        """

        super(CriticNet, self).__init__()
        self.n_feature = n_feature
        self.n_action = n_action
        self.n_conv_filter = n_conv_filter
        self.first_fc_output_dim = first_fc_output_dim
        self.hidden_fc_output_dim = hidden_fc_output_dim
        self.kernel_size = kernel_size

        # First layers
        self.throughput_conv = create_scalar_conv1d(self.n_conv_filter, self.kernel_size) # Past chunk throughput
        self.time_conv = create_scalar_conv1d(self.n_conv_filter, self.kernel_size) # Past chunk download time
        self.chunk_size_conv = create_scalar_conv1d(self.n_conv_filter, self.kernel_size) # Next chunk sizes
        self.buffer_size_fc = create_fc(1, self.first_fc_output_dim) # Current buffer size
        self.chunk_left_fc = create_fc(1, self.first_fc_output_dim) # Number of chunks left
        self.bitrate_fc = create_fc(1, self.first_fc_output_dim) # Last chunk bitrate

        hidden_layer_input_dim = \
            3 * self.first_fc_output_dim + \
            2 * self.n_conv_filter * (self.n_feature[1] - self.kernel_size + 1) + \
            self.n_conv_filter * (self.n_action - self.kernel_size + 1)

        # Hidden & output layers
        self.hidden_fc = create_fc(hidden_layer_input_dim, self.hidden_fc_output_dim)
        self.output_fc = create_fc(self.hidden_fc_output_dim, 1)

    def forward(self, x):
        """
        Critic network takes the same input as the actor network.
        """
        # First layer
        bitrate_fc_out = F.relu(self.bitrate_fc(x[:, 0: 1, -1]), inplace = True)
        buffer_size_fc_out = F.relu(self.buffer_size_fc(x[:, 1: 2, -1]), inplace = True)
        throughput_conv_out = F.relu(self.throughput_conv(x[:, 2: 3, :]), inplace = True)
        time_conv_out = F.relu(self.time_conv(x[:, 3: 4, :]), inplace = True)
        chunk_size_out = F.relu(self.chunk_size_conv(x[:, 4: 5, : self.n_action]), inplace = True)
        chunk_left_out = F.relu(self.chunk_left_fc(x[:, 5: 6, -1]), inplace = True)

        # Flatten to feed into next layer
        throughput_conv_out_flatten = throughput_conv_out.view(throughput_conv_out.shape[0], -1)
        time_conv_out_flatten = time_conv_out.view(time_conv_out.shape[0], -1)
        chunk_size_out_flatten = chunk_size_out.view(chunk_size_out.shape[0], -1)

        # Concat
        hidden_fc_in = torch.cat([
            bitrate_fc_out,
            buffer_size_fc_out,
            throughput_conv_out_flatten,
            time_conv_out_flatten,
            chunk_size_out_flatten,
            chunk_left_out,
        ], dim = 1)

        # Hidden & output layer
        hidden_fc_out = F.relu(self.hidden_fc(hidden_fc_in), inplace = True)
        return self.output_fc(hidden_fc_out).squeeze() # Shape: (batch_size, )

if __name__ == "__main__":
    pass