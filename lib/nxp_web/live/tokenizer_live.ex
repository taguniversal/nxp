defmodule NxpWeb.TokenizerLive do
  use NxpWeb, :live_view
  require Logger

  @vocab [
    # spiritual
    "god",
    "create",
    "heaven",
    "earth",
    # programming
    "language",
    "model",
    "code",
    "error",
    # sci-fi/drama
    "betray",
    "escape",
    "protect",
    "child",
    # political
    "senator",
    "reform",
    "power",
    "truth",
    # dystopian/sci-fi
    "resistance",
    "futile",
    "portal",
    "alien"
  ]

  def mount(_params, _session, socket) do
    {:ok,
     assign(socket,
       text: "",
       tokens: [],
       token_count: 0,
       embeddings: [],
       selected_head: 0,
       attention_heads: [],
       weighted_heads: [],
       projected_heads: [],
       residual_heads: [],
       normalized_heads: [],
       softmax_heads: [],
       predictions: [],
       generated_tokens: [],
       # or 4 or 8
       num_heads: 4
     )}
  end

  defp get_random_sample do
    path = Path.join(:code.priv_dir(:nxp), "tokenize_samples.json")

    with {:ok, body} <- File.read(path),
         {:ok, samples} <- Jason.decode(body),
         sample when is_binary(sample) <- Enum.random(samples) do
      {:ok, sample}
    else
      error -> {:error, inspect(error)}
    end
  end

  defp multi_head_embeddings(embeddings, num_heads) do
    for head <- 1..num_heads do
      for %{combined: vec} <- embeddings do
        # Slightly perturb each embedding to simulate per-head linear projection
        Enum.map(vec, fn x -> Float.round(x + :rand.uniform() * 0.01, 4) end)
      end
    end
  end

  defp compute_multihead_attention(embeddings, num_heads) do
    head_vectors = multi_head_embeddings(embeddings, num_heads)

    Enum.map(head_vectors, fn vectors ->
      for query <- vectors do
        row = Enum.map(vectors, fn key -> dot_product(query, key) end)
        softmax(row)
      end
    end)
  end

  defp compute_multihead_weighted_sums(mha, embeddings, num_heads) do
    head_values = multi_head_embeddings(embeddings, num_heads)

    Enum.zip(mha, head_values)
    |> Enum.map(fn {attention, values} ->
      Enum.map(attention, fn row ->
        Enum.zip(row, values)
        |> Enum.reduce(List.duplicate(0.0, length(List.first(values))), fn {weight, vec}, acc ->
          Enum.zip_with(acc, vec, fn a, b -> a + weight * b end)
        end)
        |> Enum.map(&Float.round(&1, 4))
      end)
    end)
  end

  defp add_residuals(originals, projected_heads) do
    Enum.zip(projected_heads, originals)
    |> Enum.map(fn {head_proj, head_input} ->
      Enum.zip(head_proj, head_input)
      |> Enum.map(fn {proj_vec, input_vec} ->
        Enum.zip_with(proj_vec, input_vec, fn p, i -> Float.round(p + i, 4) end)
      end)
    end)
  end

  defp layer_norm_heads(heads) do
    Enum.map(heads, fn head ->
      Enum.map(head, fn vec ->
        mean = Enum.sum(vec) / length(vec)
        squared_diffs = Enum.map(vec, fn x -> :math.pow(x - mean, 2) end)
        variance = Enum.sum(squared_diffs) / length(vec)
        stddev = :math.sqrt(variance)
        Enum.map(vec, fn x -> Float.round((x - mean) / (stddev + 1.0e-6), 4) end)
      end)
    end)
  end

  defp compute_feed_forward_heads(weighted_heads) do
    Enum.map(weighted_heads, fn head ->
      Enum.map(head, fn vec ->
        vec
        |> dense_layer1()
        |> relu()
        |> dense_layer2()
      end)
    end)
  end

  defp dense_layer1(vec) do
    # 4 → 8 dimension (expand)
    for i <- 0..7 do
      Enum.with_index(vec)
      |> Enum.reduce(0.0, fn {v, j}, acc ->
        acc + v * :math.sin((i + 1) * (j + 1) * 0.01)
      end)
      |> Float.round(4)
    end
  end

  defp dense_layer2(vec) do
    # 8 → 4 dimension (contract)
    for i <- 0..3 do
      Enum.with_index(vec)
      |> Enum.reduce(0.0, fn {v, j}, acc ->
        acc + v * :math.cos((i + 1) * (j + 1) * 0.01)
      end)
      |> Float.round(4)
    end
  end

  defp relu(vec) do
    Enum.map(vec, fn x -> if x < 0, do: 0.0, else: x end)
  end

  defp fake_embedding(token) do
    seed = :erlang.phash2(token)
    for i <- 1..4, do: :math.sin(seed * i * 0.001) |> Float.round(4)
  end

  defp positional_encoding(position) do
    for i <- 0..3 do
      :math.cos(position * (i + 1) * 0.01) |> Float.round(4)
    end
  end

  defp add_vectors(a, b) do
    Enum.zip_with(a, b, fn x, y -> Float.round(x + y, 4) end)
  end

  defp dot_product(a, b) do
    Enum.zip_with(a, b, fn x, y -> x * y end) |> Enum.sum()
  end

  defp softmax(row) do
    max_val = Enum.max(row)
    exps = Enum.map(row, fn x -> :math.exp(x - max_val) end)
    sum = Enum.sum(exps)
    Enum.map(exps, &Float.round(&1 / sum, 4))
  end

  defp compute_attention_matrix(embeddings) do
    vectors = Enum.map(embeddings, & &1.combined)

    for query <- vectors do
      row = Enum.map(vectors, fn key -> dot_product(query, key) end)
      softmax(row)
    end
  end

  def handle_event("tokenize", %{"input" => %{"text" => text}}, socket) do
    tokens = String.split(text, ~r/\s+/, trim: true)

    embeddings =
      tokens
      |> Enum.with_index()
      |> Enum.map(fn {token, idx} ->
        embed = fake_embedding(token)
        pos = positional_encoding(idx)
        encoded = add_vectors(embed, pos)

        %{
          token: token,
          position: idx,
          embedding: embed,
          pos_encoding: pos,
          combined: encoded
        }
      end)

    attention_heads = compute_multihead_attention(embeddings, socket.assigns.num_heads)

    weighted_heads =
      compute_multihead_weighted_sums(attention_heads, embeddings, socket.assigns.num_heads)

    projected_heads = compute_feed_forward_heads(weighted_heads)
    residual_heads = add_residuals(weighted_heads, projected_heads)
    normalized_heads = layer_norm_heads(residual_heads)
    softmax_heads = compute_softmax_predictions(normalized_heads, tokens)
    predictions = get_top_predictions(softmax_heads)

    {:noreply,
     assign(socket,
       text: text,
       tokens: Enum.with_index(tokens),
       token_count: length(tokens),
       embeddings: embeddings,
       attention_heads: attention_heads,
       weighted_heads: weighted_heads,
       projected_heads: projected_heads,
       residual_heads: residual_heads,
       normalized_heads: normalized_heads,
       softmax_heads: softmax_heads,
       predictions: predictions
     )}
  end

  defp generate_tokens(socket) do
    initial_tokens = socket.assigns.tokens |> Enum.map(fn {tok, _i} -> tok end)

    # Only use previously generated tokens if they exist
    context =
      case socket.assigns.generated_tokens do
        nil -> initial_tokens
        [] -> initial_tokens
        prior -> Enum.map(prior, fn {tok, _i} -> tok end)
      end

    new_tokens = do_generate(context, 10)

    updated =
      new_tokens
      |> Enum.drop(length(context)) # 🧠 Only the newly generated ones
      |> Enum.with_index()

    assign(socket, :generated_tokens, context ++ updated)
  end

  def handle_event("generate", _params, socket) do
    {:noreply, generate_tokens(socket)}
  end


  defp get_last_output_vector(token, assigns) do
    tokens = assigns.tokens
    idx = Enum.find_index(tokens, fn {t, _} -> t == token end) || 0

    assigns.normalized_heads
    |> List.first()
    |> Enum.at(idx)
  end

  def handle_event("change_head", %{"head" => head_str}, socket) do
    head = String.to_integer(head_str)
    Logger.info("Changing head to #{head}")
    {:noreply, assign(socket, selected_head: head)}
  end

  def handle_event("load_sample", _params, socket) do
    case get_random_sample() do
      {:ok, sample} ->
        send(self(), {:tokenize_sample, sample})
        {:noreply, assign(socket, text: sample)}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to load sample: #{reason}")}
    end
  end

  def handle_info({:tokenize_sample, text}, socket) do
    tokens = String.split(text, ~r/\s+/, trim: true)

    embeddings =
      tokens
      |> Enum.with_index()
      |> Enum.map(fn {token, idx} ->
        embed = fake_embedding(token)
        pos = positional_encoding(idx)
        encoded = add_vectors(embed, pos)

        %{
          token: token,
          position: idx,
          embedding: embed,
          pos_encoding: pos,
          combined: encoded
        }
      end)

    attention_heads = compute_multihead_attention(embeddings, socket.assigns.num_heads)

    weighted_heads =
      compute_multihead_weighted_sums(attention_heads, embeddings, socket.assigns.num_heads)

    projected_heads = compute_feed_forward_heads(weighted_heads)
    residual_heads = add_residuals(weighted_heads, projected_heads)
    normalized_heads = layer_norm_heads(residual_heads)
    softmax_heads = compute_softmax_predictions(normalized_heads, tokens)
    predictions = get_top_predictions(softmax_heads)

    {:noreply,
     assign(socket,
       text: text,
       tokens: Enum.with_index(tokens),
       token_count: length(tokens),
       embeddings: embeddings,
       attention_heads: attention_heads,
       weighted_heads: weighted_heads,
       projected_heads: projected_heads,
       residual_heads: residual_heads,
       normalized_heads: normalized_heads,
       softmax_heads: softmax_heads,
       predictions: predictions
     )}
  end

  defp compute_weighted_sums(attention, embeddings) do
    values = Enum.map(embeddings, & &1.combined)

    Enum.map(attention, fn row ->
      Enum.zip(row, values)
      |> Enum.reduce(List.duplicate(0.0, length(List.first(values))), fn {weight, vec}, acc ->
        Enum.zip_with(acc, vec, fn a, b -> a + weight * b end)
      end)
      |> Enum.map(&Float.round(&1, 4))
    end)
  end

  defp fake_vocab_embeddings(tokens) do
    Enum.map(@vocab, fn vocab_word ->
      seed = :erlang.phash2(vocab_word <> Enum.join(tokens, ""))
      for i <- 1..4, do: :math.sin(seed * i * 0.001) |> Float.round(4)
    end)
  end

  defp compute_softmax_predictions(normalized_heads, tokens) do
    vocab = fake_vocab_embeddings(tokens)

    Enum.map(normalized_heads, fn head ->
      Enum.map(head, fn vec ->
        logits = Enum.map(vocab, fn vocab_vec -> dot_product(vec, vocab_vec) end)
        softmax(logits)
      end)
    end)
  end

  defp get_top_predictions(softmax_heads) do
    Enum.map(softmax_heads, fn head ->
      Enum.map(head, fn probs ->
        {max_val, max_idx} = Enum.with_index(probs) |> Enum.max_by(fn {p, _} -> p end)
        %{token: Enum.at(@vocab, max_idx), confidence: Float.round(max_val, 4)}
      end)
    end)
  end

  defp strip_token({token, _}), do: token

  @max_tokens 12

defp generate_autoregressive_sequence(seed_tokens \\ ["Every"]) do
  do_generate(seed_tokens, @max_tokens)
end

defp do_generate(tokens, 0), do: tokens

defp do_generate(tokens, remaining) do
  context = tokens

  vocab = fake_vocab_embeddings(context)

  # Add slight randomness to context vector
  vec =
    for i <- 1..4 do
      seed = :erlang.phash2({Enum.join(context), remaining})
      base = :math.sin(seed * i * 0.001)
      jitter = (:rand.uniform() - 0.5) * 0.01
      Float.round(base + jitter, 4)
    end

  logits = Enum.map(vocab, &dot_product(vec, &1))
  probs = softmax(logits)

  {predicted_token, _confidence} =
    Enum.zip(@vocab, probs)
    |> sample_from_distribution()


  do_generate(tokens ++ [predicted_token], remaining - 1)
end

defp sample_from_distribution(pairs) do
  r = :rand.uniform()
  Enum.reduce_while(pairs, 0.0, fn {token, prob}, acc ->
    if r < acc + prob do
      {:halt, {token, prob}}
    else
      {:cont, acc + prob}
    end
  end)
end


end
