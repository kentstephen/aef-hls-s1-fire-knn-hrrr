import "maplibre-gl/dist/maplibre-gl.css";
import "./styles.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.js";
import Triage from "./triage/Triage.js";

// the triage dashboard by default; ?app=fuel opens the earlier fuel map
const Root = new URLSearchParams(location.search).get("app") === "fuel" ? App : Triage;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
