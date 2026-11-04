import type { MapboxOverlayProps } from "@deck.gl/mapbox";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { useControl } from "react-map-gl/maplibre";

/** Inlined from deck.gl-raster/examples/_shared. */
export function DeckGlOverlay({ overlayRef, ...props }: MapboxOverlayProps & { overlayRef?: { current: MapboxOverlay | null } }) {
  const overlay = useControl<MapboxOverlay>(() => new MapboxOverlay(props));
  overlay.setProps(props);
  // the play loop sets the layers between renders through this
  if (overlayRef) overlayRef.current = overlay;
  // debug hook for the headless probe
  (globalThis as unknown as { __overlay?: MapboxOverlay }).__overlay = overlay;
  return null;
}
