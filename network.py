import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np_to_th(x):
    """
    Convierte np.ndarray -> torch.Tensor en DEVICE.
    Mantiene shape (N, -1) para compatibilidad; el reshape a (N,2,F) lo hace el forward().
    """
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float32).to(DEVICE).reshape(n_samples, -1)


class Net(nn.Module):
    """
    Arquitectura:
      Conv1D + GlobalAveragePooling + 3*(Linear + ReLU) + Linear(out)

    Input esperado:
      - X con shape (N, 2*F) donde la primera mitad es I y la segunda mitad Q
      - Internamente se reinterpreta como (N, C=2, L=F)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=128,          
        epochs=1000,
        loss=nn.MSELoss(),
        lr=1e-3,
        conv_channels=64,     
        kernel_size=9,        
    ) -> None:
        super().__init__()

        if input_dim % 2 != 0:
            raise ValueError("input_dim debe ser par (I || Q).")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.F = self.input_dim // 2

        self.epochs = epochs
        self.loss = loss
        self.lr = lr
        self.n_units = int(n_units)

        k = int(kernel_size)
        if k < 1 or k % 2 == 0:
            raise ValueError("kernel_size debe ser impar y >= 1 para padding 'same' simple.")
        padding = k // 2

        # Conv1D: 2 canales (I y Q)
        self.conv = nn.Conv1d(
            in_channels=2,
            out_channels=int(conv_channels),
            kernel_size=k,
            padding=padding,
            bias=True,
        )
        self.act = nn.ReLU()

        # Global Average Pooling: (N, C, L) -> (N, C, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 3*(Linear + ReLU)
        self.fc1 = nn.Linear(int(conv_channels), self.n_units)
        self.fc2 = nn.Linear(self.n_units, self.n_units)
        self.fc3 = nn.Linear(self.n_units, self.n_units)

        # salida
        self.out = nn.Linear(self.n_units, self.output_dim)

    def forward(self, x):
        # x: (N, 2F) -> (N, 2, F)
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input shape incompatible: esperaba (N, {self.input_dim}), tengo (N, {x.shape[1]})."
            )

        I = x[:, : self.F]
        Q = x[:, self.F :]

        x = torch.stack([I, Q], dim=1)  # (N, 2, F)

        h = self.act(self.conv(h := x))
        h = self.gap(h).squeeze(-1)     # (N, conv_channels)

        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))

        return self.out(h)

    def fit(self, X, y):
        Xt = np_to_th(X)
        yt = np_to_th(y)

        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        self.train()

        losses = []
        log_every = max(1, self.epochs // 10)

        for ep in range(self.epochs):
            optimiser.zero_grad()
            outputs = self.forward(Xt)
            loss = self.loss(yt, outputs)
            loss.backward()
            optimiser.step()

            losses.append(float(loss.item()))
            if ep % log_every == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.6f}")

        return losses

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            out = self.forward(np_to_th(X))
        return out.detach().cpu().numpy()


class NetDiscovery(Net):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=128,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=1e-3,
        conv_channels=64,
        kernel_size=9,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            n_units=n_units,
            epochs=epochs,
            loss=loss,
            lr=lr,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
        )

        # Mantengo tu parámetro extra por compatibilidad con lo que ya tenías
        self.r = nn.Parameter(data=torch.tensor([0.0], device=DEVICE))

