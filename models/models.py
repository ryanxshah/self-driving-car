from pathlib import Path

import torch
import torch.nn as nn

DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * n_track * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_waypoints * 2)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """

        # Track_boundary: A (b, 2, 10, 2) tensor
        # Gets flattened to (b, 40)
        x = torch.stack((track_left, track_right), dim=1)
        flattened_waypoint_preds = self.model(x)
        # Reshape outputs
        waypoint_preds = flattened_waypoint_preds.view(-1, self.n_waypoints, 2)
        # Return (128, 3, 2) tensor
        return waypoint_preds


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        self.query_embed = nn.Embedding(n_waypoints, d_model) # [3, 64]


        self.flatten_inputs = nn.Flatten()
        self.input_proj = nn.Linear(1, d_model)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, batch_first=True
        )

        self.latent_transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, batch_first=True
            )
            for _ in range(2)
        ])

        self.output_head = nn.Linear(d_model, 2)




    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        
        
        inputs = torch.stack((track_left, track_right), dim=1) # [B, 2, 10, 2]
        inputs = self.flatten_inputs(inputs) # [B, 40]
        inputs = inputs.unsqueeze(-1)
        byte_array = self.input_proj(inputs)

        latent_array = self.query_embed.weight.unsqueeze(0).expand(inputs.size(0), -1, -1)  # Shape: (b, n_waypoints, d_model)

        # cross attn
        latent_array, _ = self.cross_attention(query=latent_array, key=byte_array, value=byte_array)

        for layer in self.latent_transformer:
            latent_array = layer(latent_array)

        waypoints = self.output_head(latent_array)

        return waypoints


        


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=2, padding=(9-1)//2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=(3-1)//2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=(3-1)//2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=(3-1)//2),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, n_waypoints * 2, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )


    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """

        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        preds = self.model(x).view(-1, self.n_waypoints, 2)
        return preds


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
