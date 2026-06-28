def risk_ui(risk):
    if risk == "LOW":
        return {"label": "LOW RISK", "color": "#22c55e", "bg": "#ecfdf5", "emoji": "🟢"}
    elif risk == "MEDIUM":
        return {"label": "MEDIUM RISK", "color": "#f59e0b", "bg": "#fffbeb", "emoji": "🟠"}
    else:
        return {"label": "HIGH RISK", "color": "#ef4444", "bg": "#fef2f2", "emoji": "🔴"}
