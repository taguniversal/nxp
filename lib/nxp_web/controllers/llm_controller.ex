defmodule NxpWeb.LLMController do
  use NxpWeb, :controller
  require Logger

  @default_model "gemma3:4b"

  def generate(conn, %{"prompt" => prompt}) do
    body = Jason.encode!(%{
      model: @default_model,
      prompt: prompt,
      stream: false
    })

    req = Finch.build(:post, "http://localhost:11434/api/generate", [
      {"content-type", "application/json"}
    ], body)

    case Finch.request(req, Nxp.Finch, receive_timeout: 320_000) do
      {:ok, %Finch.Response{status: 200, body: body}} ->
        Logger.info("Ollama raw body: #{inspect(body)}")
        chunks = String.split(body, "\n", trim: true)

        text =
          chunks
          |> Enum.map(fn line ->
            case Jason.decode(line) do
              %{"response" => part} -> part
              _ -> ""
            end
          end)
          |> Enum.join("")

        json(conn, %{response: text})

      {:ok, %Finch.Response{status: code}} ->
        json(conn, %{error: "Ollama error: #{code}"})

      {:error, reason} ->
        Logger.error("Finch error: #{inspect(reason)}")
        json(conn, %{error: "Finch error: #{inspect(reason)}"})
    end
  end

  def generate(conn, _params) do
    json(conn, %{error: "Missing prompt parameter"})
  end
end
