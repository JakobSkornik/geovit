import json
import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=0.1, heatmap_file="heatmap.json"):
        super(WeightedMSELoss, self).__init__()
        self.hm_file = heatmap_file
        self.alpha = alpha
        self.heatmap = self.__load_heatmap()

    def __load_heatmap(self):
        with open(self.hm_file, "r") as fp:
            return json.load(fp)

    def get_heatmap_weight(self, coords):
        # Convert the denormalized coordinates to grid indices
        x, y = (coords[..., 0] // 0.9).long(), (coords[..., 1] // 1.8).long()

        # Create an empty tensor to store weights
        weights = torch.zeros(coords.shape[0], device=coords.device)

        for i, (xi, yi) in enumerate(zip(x, y)):
            key = f"{xi.item()}_{yi.item()}"
            # Assign the corresponding value from the heatmap or a default value of 0.0 if the key doesn't exist
            weights[i] = self.heatmap.get(key, 0.0)

        return weights

    def forward(self, predictions, labels):
        # Calculate MAE error
        mse_error = abs(predictions - labels)

        # Get the heatmap weights for predictions
        weights = self.get_heatmap_weight(denormalize_coordinates_torch(predictions))

        # Compute weighted MSE error using the heatmap weights
        weighted_mse = mse_error + mse_error * (self.alpha - self.alpha * weights.unsqueeze(-1))

        return weighted_mse.mean()

def denormalize_coordinates_torch(coords):
    latitude, longitude = coords[..., 0], coords[..., 1]
    latitude = latitude * 180.0 - 90.0
    longitude = longitude * 360.0 - 180.0
    return torch.stack([latitude, longitude], dim=-1)
