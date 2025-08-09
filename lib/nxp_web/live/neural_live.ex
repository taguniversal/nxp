defmodule NxpWeb.NeuralLive do
  use NxpWeb, :live_view
  require Logger

  @impl true
  def mount(_params, _session, socket) do
    {:ok,
     socket
     |> assign_new(:seed, fn -> :rand.uniform(1_000_000) end)
     |> assign(
       # UI / hyperparams
       activation: "relu",
       learning_rate: 0.05,
       layers: [2, 8, 1],          # input, hidden, output
       epoch: 0,

       # data
       dataset: generate_toy_data(160),
       predictions: [],

       # network state (placeholders for now)
       weights: %{},               # %{layer_index => matrix}
       biases: %{},                # %{layer_index => vector}
       gradients: %{},             # %{layer_index => matrix}
       loss: nil,
       loss_history: [],
       params: nil,          # 👈 holds %{w1,b1,w2,b2}
       boundary: []
     )}
  end

  @impl true
  def handle_event("randomize_data", _params, socket) do
    {:noreply, assign(socket, dataset: generate_toy_data(80), predictions: [], epoch: 0, loss_history: [], loss: nil)}
  end


  def handle_event("reset_model", _params, socket) do
    hidden = Enum.at(socket.assigns.layers, 1)
    params = init_params(hidden)
    boundary = compute_boundary(params, socket.assigns.activation)
    {:noreply,
     assign(socket,
       params: params,
       predictions: [],
       weights: %{},
       biases: %{},
       gradients: %{},
       epoch: 0,
       loss: nil,
       loss_history: [],
       boundary: boundary
     )}
  end

  def handle_event("change_hparam", %{"field" => "activation", "value" => act}, socket) do
    {:noreply, assign(socket, activation: act)}
  end

  def handle_event("change_hparam", %{"field" => "learning_rate", "value" => lr}, socket) do
    lr =
      case Float.parse(lr) do
        {v, _} -> v
        :error -> socket.assigns.learning_rate
      end

    {:noreply, assign(socket, learning_rate: lr)}
  end

  def handle_event("change_layers", %{"hidden" => hidden_str}, socket) do
    hidden =
      case Integer.parse(hidden_str) do
        {v, _} when v > 0 -> v
        _ -> Enum.at(socket.assigns.layers, 1)
      end

    {:noreply, assign(socket, layers: [2, hidden, 1])}
  end

  def handle_event("forward", _params, socket) do
    params = socket.assigns.params || init_params(Enum.at(socket.assigns.layers, 1))
    {preds, loss} = predict_and_loss(params, socket.assigns.dataset, socket.assigns.activation)
    boundary = compute_boundary(params, socket.assigns.activation)   # 👈 add
    {:noreply, assign(socket, params: params, predictions: preds, loss: loss, boundary: boundary)}
  end

  def handle_event("train_step", _params, socket) do
    params = socket.assigns.params || init_params(Enum.at(socket.assigns.layers, 1))
    {params2, loss} =
      train_one_epoch(params, socket.assigns.dataset, socket.assigns.learning_rate, socket.assigns.activation)

    boundary = compute_boundary(params2, socket.assigns.activation)  # 👈 add
    {preds2, _} = predict_and_loss(params2, socket.assigns.dataset, socket.assigns.activation)
    {:noreply,
     assign(socket,
       params: params2,
       epoch: socket.assigns.epoch + 1,
       loss: loss,
       loss_history: socket.assigns.loss_history ++ [loss],
       predictions: preds2,
       boundary: boundary
     )}
  end



  # ---------- Helpers (toy placeholders) ----------
# ---------- MLP params ----------
defp init_params(h) do
  w1 =
    for _ <- 1..h do
      for _ <- 1..2 do
        rand_small()
      end
    end

  b1 = for _ <- 1..h, do: 0.0
  w2 = for _ <- 1..h, do: rand_small()

  %{
    w1: w1,   # h x 2
    b1: b1,   # h
    w2: w2,   # 1 x h
    b2: 0.0   # scalar
  }
end

defp rand_small, do: (:rand.uniform() - 0.5) * 0.2  # symmetric, small

# ---------- Activations ----------
defp act("relu", x), do: if(x < 0, do: 0.0, else: x)
defp act("sigmoid", x), do: 1.0 / (1.0 + :math.exp(-x))
defp act("tanh", x), do: :math.tanh(x)
defp act(_, x), do: act("relu", x)

defp actprime("relu", x), do: if(x <= 0.0, do: 0.0, else: 1.0)
defp actprime("sigmoid", x) do
  s = 1.0 / (1.0 + :math.exp(-x))
  s * (1.0 - s)
end
defp actprime("tanh", x) do
  t = :math.tanh(x)
  1.0 - t * t
end
defp actprime(_, x), do: actprime("relu", x)

# ---------- Forward (one sample) ----------
defp forward_one(%{w1: w1, b1: b1, w2: w2, b2: b2}, {x, y}, activation) do
  # x = {x1, x2}
  {x1, x2} = x

  # layer 1: z1_j = w1_j1*x1 + w1_j2*x2 + b1_j
  z1 =
    Enum.map(Enum.zip(w1, b1), fn {row, bj} ->
      [w11, w12] = row
      w11 * x1 + w12 * x2 + bj
    end)

  a1 = Enum.map(z1, &act(activation, &1))

  # output: z2 = sum_j w2_j*a1_j + b2; yhat = sigmoid(z2)
  z2 = Enum.zip(w2, a1) |> Enum.reduce(b2, fn {wj, aj}, acc -> acc + wj * aj end)
  yhat = 1.0 / (1.0 + :math.exp(-z2))

  cache = %{x: x, y: y, z1: z1, a1: a1, z2: z2, yhat: yhat}
  {yhat, cache}
end

# ---------- Backward (one sample) ----------
# Binary cross-entropy with sigmoid output -> dL/dz2 = (yhat - y)
defp backward_one(params, %{x: {x1, x2}, y: y, z1: z1, a1: a1, yhat: yhat}, activation) do
  %{w2: w2} = params
  dz2 = yhat - y

  # dW2_j = dz2 * a1_j ; db2 = dz2
  dW2 = Enum.map(a1, fn aj -> dz2 * aj end)
  db2 = dz2

  # hidden layer
  # da1_j = w2_j * dz2 ; dz1_j = da1_j * act'(z1_j)
  dz1 =
    Enum.zip(w2, z1)
    |> Enum.map(fn {w2j, z1j} ->
      (w2j * dz2) * actprime(activation, z1j)
    end)

  # dW1_jk = dz1_j * x_k ; db1_j = dz1_j
  dW1 =
    Enum.map(dz1, fn dz1j -> [dz1j * x1, dz1j * x2] end)

  db1 = dz1

  %{dW1: dW1, db1: db1, dW2: dW2, db2: db2}
end
# ---------- SGD update (averaged over dataset) ----------
defp train_one_epoch(params, dataset, lr, activation) do
  # build init parts outside the map (avoids ambiguity) + clearer
  dW1_init = for _ <- params.w1, do: [0.0, 0.0]
  db1_init = for _ <- params.b1, do: 0.0
  dW2_init = for _ <- params.w2, do: 0.0

  init = %{
    dW1: dW1_init,
    db1: db1_init,
    dW2: dW2_init,
    db2: 0.0,
    loss_sum: 0.0
  }

  n = max(length(dataset), 1)

  acc =
    Enum.reduce(dataset, init, fn p, acc_in ->
      x = {p.x, p.y}
      y = p.label * 1.0

      {yhat, cache} = forward_one(params, {x, y}, activation)

      # BCE loss (clamped)
      yh = min(max(yhat, 1.0e-6), 1.0 - 1.0e-6)
      loss = -(y * :math.log(yh) + (1.0 - y) * :math.log(1.0 - yh))

      grads = backward_one(params, cache, activation)

      %{
        dW1: add_mats(acc_in.dW1, grads.dW1),
        db1: add_vec(acc_in.db1, grads.db1),
        dW2: add_vec(acc_in.dW2, grads.dW2),
        db2: acc_in.db2 + grads.db2,
        loss_sum: acc_in.loss_sum + loss
      }
    end)

  # average
  scale = 1.0 / n
  dW1 = scale_mat(acc.dW1, scale)
  db1 = Enum.map(acc.db1, &(&1 * scale))
  dW2 = Enum.map(acc.dW2, &(&1 * scale))
  db2 = acc.db2 * scale
  avg_loss = Float.round(acc.loss_sum * scale, 4)

  # update
  params2 = %{
    w1: sub_mats(params.w1, scale_mat(dW1, lr)),
    b1: Enum.zip(params.b1, db1) |> Enum.map(fn {b, g} -> b - lr * g end),
    w2: Enum.zip(params.w2, dW2) |> Enum.map(fn {w, g} -> w - lr * g end),
    b2: params.b2 - lr * db2
  }

  {params2, avg_loss}
end

# ---------- Prediction & loss over dataset ----------
defp predict_and_loss(params, dataset, activation) do
  {probs, loss_sum} =
    Enum.reduce(dataset, {[], 0.0}, fn p, {acc, lsum} ->
      {yhat, _cache} = forward_one(params, {{p.x, p.y}, p.label * 1.0}, activation)
      yh = min(max(yhat, 1.0e-6), 1.0 - 1.0e-6)
      loss = -(p.label * :math.log(yh) + (1 - p.label) * :math.log(1.0 - yh))
      {[Float.round(yhat, 4) | acc], lsum + loss}
    end)

  n = max(length(dataset), 1)
  {Enum.reverse(probs), Float.round(loss_sum / n, 4)}
end

# ---------- tiny vector/matrix helpers ----------
defp add_vec(a, b), do: Enum.zip(a, b) |> Enum.map(fn {x, y} -> x + y end)
defp add_mats(a, b), do: Enum.zip(a, b) |> Enum.map(fn {ra, rb} -> add_vec(ra, rb) end)

defp sub_mats(a, b), do: Enum.zip(a, b) |> Enum.map(fn {ra, rb} -> sub_vec(ra, rb) end)
defp sub_vec(a, b), do: Enum.zip(a, b) |> Enum.map(fn {x, y} -> x - y end)

defp scale_mat(m, s), do: Enum.map(m, fn row -> Enum.map(row, &(&1 * s)) end)

  defp generate_toy_data(n) do
    # Returns [%{x: float, y: float, label: 0|1}]
    for _ <- 1..n do
      x = :rand.uniform() * 2 - 1
      y = :rand.uniform() * 2 - 1
      label = if x * x + y * y > 0.5, do: 1, else: 0
      %{x: Float.round(x, 3), y: Float.round(y, 3), label: label}
    end
  end

  defp forward_demo(points, activation) do
    Enum.map(points, fn p ->
      z = p.x * 0.8 + p.y * -0.6
      a =
        case activation do
          "relu" -> if z < 0, do: 0.0, else: z
          "tanh" -> :math.tanh(z)
          _ -> 1.0 / (1.0 + :math.exp(-z)) # sigmoid
        end

      prob = 1.0 / (1.0 + :math.exp(-a))
      Float.round(prob, 4)
    end)
  end

  defp demo_loss(points, preds) do
    # BCE-ish toy loss vs. labels
    labels = Enum.map(points, & &1.label)
    pairs = Enum.zip(labels, preds)

    avg =
      pairs
      |> Enum.map(fn {y, p} ->
        # clamp
        p = min(max(p, 1.0e-6), 1 - 1.0e-6)
        -(y * :math.log(p) + (1 - y) * :math.log(1 - p))
      end)
      |> Enum.sum()
      |> Kernel./(length(pairs))

    Float.round(avg, 4)
  end

  # --- Viz helpers -------------------------------------------------------------

# Map from data space [-1, 1] to SVG pixel space with padding
def to_screen({x, y}, size \\ 320, pad \\ 16) do
  # flip y for SVG
  sx = pad + (x + 1.0) / 2.0 * (size - 2 * pad)
  sy = pad + (1.0 - (y + 1.0) / 2.0) * (size - 2 * pad)
  {Float.round(sx, 2), Float.round(sy, 2)}
end

def label_color(0), do: "#1d4ed8"   # blue (class 0)
def label_color(1), do: "#dc2626"   # red  (class 1)

# Green-ish heat based on probability (0..1)
# Blue for p<0.5, Red for p>0.5 with intensity away from 0.5
def prob_color(p) do
  t = p |> min(1.0) |> max(0.0)
  if t >= 0.5 do
    alpha = (t - 0.5) * 2
    "rgba(220, 38, 38, #{Float.round(alpha, 3)})"    # red-600
  else
    alpha = (0.5 - t) * 2
    "rgba(29, 78, 216, #{Float.round(alpha, 3)})"    # blue-700
  end
end

# Blue for p<0.5, Red for p>0.5; return {r,g,b,alpha}
def prob_rgba(p) do
  t = p |> min(1.0) |> max(0.0)
  if t >= 0.5 do
    alpha = (t - 0.5) * 2.0
    {220, 38, 38, Float.round(alpha, 3)}      # red-600
  else
    alpha = (0.5 - t) * 2.0
    {29, 78, 216, Float.round(alpha, 3)}      # blue-700
  end
end



# Make a coarse prediction grid for a decision surface
defp compute_boundary(%{w1: _w1} = params, activation, n \\ 36) do
  xs = for i <- 0..n, do: -1.0 + 2.0 * (i / n)
  ys = for j <- 0..n, do: -1.0 + 2.0 * (j / n)

  for x <- xs, y <- ys do
    {p, _} = forward_one(params, {{x, y}, 0.0}, activation)
    %{x: x, y: y, p: p}
  end
end

end
