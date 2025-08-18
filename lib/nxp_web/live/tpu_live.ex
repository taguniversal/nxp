defmodule NxpWeb.TPULive do
  use NxpWeb, :live_view

  @impl true
  def mount(_params, _session, socket) do
    {:ok, assign(socket, :topology, "mesh")}
  end

  @impl true
  def handle_event("select_topology", %{"topology" => topology}, socket) do
    {:noreply, assign(socket, :topology, topology)}
  end

end
