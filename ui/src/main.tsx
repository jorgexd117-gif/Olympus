import React, { useState } from "react";
import ReactDOM from "react-dom/client";

import Landing from "./Landing";
import App from "./App";
import "./styles.css";

function Root() {
  const [inApp, setInApp] = useState(false);

  if (inApp) {
    return <App />;
  }
  return <Landing onEnterApp={() => setInApp(true)} />;
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
