import json
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


st.set_page_config(
    page_title="TFM License Plate Dashboard",
    page_icon="🚘",
    layout="wide",
)


API_BASE_URL = st.secrets["API_BASE_URL"].rstrip("/")
COGNITO_TOKEN_URL = st.secrets["COGNITO_TOKEN_URL"]
ADMIN_CLIENT_ID = st.secrets["ADMIN_CLIENT_ID"]
ADMIN_CLIENT_SECRET = st.secrets["ADMIN_CLIENT_SECRET"]
ADMIN_SCOPES = st.secrets.get(
    "ADMIN_SCOPES",
    "tfm-api/admin-watchlist tfm-api/dashboard-read",
)


@st.cache_data(ttl=3300)
def get_admin_token() -> str:
    response = requests.post(
        COGNITO_TOKEN_URL,
        auth=(ADMIN_CLIENT_ID, ADMIN_CLIENT_SECRET),
        data={
            "grant_type": "client_credentials",
            "scope": ADMIN_SCOPES,
        },
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=15,
    )

    response.raise_for_status()
    return response.json()["access_token"]


def api_request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    token = get_admin_token()

    url = f"{API_BASE_URL}{path}"

    response = requests.request(
        method=method,
        url=url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        params=params,
        timeout=20,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Error API {method} {path}: "
            f"{response.status_code} {response.text}"
        )

    if not response.text:
        return {}

    return response.json()


def to_dataframe(items: list[dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()

    return pd.json_normalize(items)


def page_summary() -> None:
    st.title("Resumen general")

    data = api_request("GET", "/dashboard/summary")
    totals = data.get("totals", {})

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Imágenes procesadas", totals.get("total_images_processed", 0))
    col2.metric("Detecciones", totals.get("total_detections", 0))
    col3.metric("Éxito OCR", f"{totals.get('ocr_success_rate', 0) * 100:.1f}%")
    col4.metric("Alertas abiertas", totals.get("open_alerts", 0))

    col5, col6, col7, col8 = st.columns(4)

    col5.metric("Matches watchlist", totals.get("total_watchlist_matches", 0))
    col6.metric("Alertas totales", totals.get("total_alerts", 0))
    col7.metric("Raspberry registradas", totals.get("registered_devices", 0))
    col8.metric("Raspberry activas", totals.get("active_devices", 0))

    st.subheader("Últimos summaries de inferencia")
    latest_runs = data.get("latest_inference_runs", [])
    df_runs = to_dataframe(latest_runs)
    st.dataframe(df_runs, use_container_width=True)

    st.subheader("Últimas alertas")
    latest_alerts = data.get("latest_alerts", [])
    df_alerts = to_dataframe(latest_alerts)
    st.dataframe(df_alerts, use_container_width=True)


def page_visualizations() -> None:
    st.title("Visualización de métricas")

    summary_data = api_request("GET", "/dashboard/summary")
    totals = summary_data.get("totals", {})

    alerts_data = api_request("GET", "/dashboard/alerts", params={"limit": 200})
    alerts = alerts_data.get("items", [])

    devices_data = api_request("GET", "/dashboard/devices")
    devices = devices_data.get("items", [])

    # ======================================================
    # 1. KPIs principales
    # ======================================================
    st.subheader("KPIs principales")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Imágenes procesadas",
        totals.get("total_images_processed", 0),
    )

    col2.metric(
        "Detecciones totales",
        totals.get("total_detections", 0),
    )

    col3.metric(
        "Éxito OCR",
        f"{totals.get('ocr_success_rate', 0) * 100:.1f}%",
    )

    col4.metric(
        "Alertas abiertas",
        totals.get("open_alerts", 0),
    )

    col5, col6, col7, col8 = st.columns(4)

    col5.metric(
        "Matches watchlist",
        totals.get("total_watchlist_matches", 0),
    )

    col6.metric(
        "Alertas totales",
        totals.get("total_alerts", 0),
    )

    col7.metric(
        "Raspberry registradas",
        totals.get("registered_devices", 0),
    )

    col8.metric(
        "Raspberry activas",
        totals.get("active_devices", 0),
    )

    st.divider()

    # ======================================================
    # Preparación de datos de alertas
    # ======================================================
    df_alerts = to_dataframe(alerts)

    if df_alerts.empty:
        st.info("Todavía no hay alertas registradas para generar gráficas.")
    else:
        if "status" not in df_alerts.columns:
            df_alerts["status"] = "unknown"

        if "plate_text" not in df_alerts.columns:
            df_alerts["plate_text"] = "unknown"

        if "device_id" not in df_alerts.columns:
            df_alerts["device_id"] = "unknown"

        # ==================================================
        # 2. Estados de alertas
        # ==================================================
        st.subheader("Distribución de estados de alertas")

        status_counts = (
            df_alerts["status"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .reset_index()
        )
        status_counts.columns = ["status", "count"]

        col1, col2 = st.columns(2)

        with col1:
            fig_status_pie = px.pie(
                status_counts,
                names="status",
                values="count",
                title="Alertas por estado",
                hole=0.35,
            )
            st.plotly_chart(fig_status_pie, use_container_width=True)

        with col2:
            fig_status_bar = px.bar(
                status_counts,
                x="status",
                y="count",
                title="Número de alertas por estado",
                text="count",
            )
            fig_status_bar.update_layout(
                xaxis_title="Estado",
                yaxis_title="Número de alertas",
            )
            st.plotly_chart(fig_status_bar, use_container_width=True)

        st.divider()

        # ==================================================
        # 3. Alertas por matrícula
        # ==================================================
        st.subheader("Alertas por matrícula")

        plate_counts = (
            df_alerts["plate_text"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .reset_index()
            .head(10)
        )
        plate_counts.columns = ["plate_text", "count"]

        fig_plate = px.bar(
            plate_counts,
            x="count",
            y="plate_text",
            orientation="h",
            title="Top 10 matrículas con más alertas",
            text="count",
        )
        fig_plate.update_layout(
            xaxis_title="Número de alertas",
            yaxis_title="Matrícula",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_plate, use_container_width=True)

        st.divider()

        # ==================================================
        # 4. Alertas por Raspberry
        # ==================================================
        st.subheader("Alertas por Raspberry")

        device_alert_counts = (
            df_alerts["device_id"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .reset_index()
        )
        device_alert_counts.columns = ["device_id", "count"]

        fig_device_alerts = px.bar(
            device_alert_counts,
            x="device_id",
            y="count",
            title="Alertas detectadas por dispositivo",
            text="count",
        )
        fig_device_alerts.update_layout(
            xaxis_title="Dispositivo",
            yaxis_title="Número de alertas",
        )
        st.plotly_chart(fig_device_alerts, use_container_width=True)

    st.divider()

    # ======================================================
    # 5. Dispositivos activos
    # ======================================================
    st.subheader("Estado de dispositivos")

    df_devices = to_dataframe(devices)

    if df_devices.empty:
        st.info("Todavía no hay dispositivos registrados.")
    else:
        if "status" not in df_devices.columns:
            df_devices["status"] = "unknown"

        if "device_id" not in df_devices.columns:
            df_devices["device_id"] = "unknown"

        device_status_counts = (
            df_devices["status"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .reset_index()
        )
        device_status_counts.columns = ["status", "count"]

        col1, col2 = st.columns([1, 2])

        with col1:
            fig_devices_status = px.pie(
                device_status_counts,
                names="status",
                values="count",
                title="Dispositivos por estado",
                hole=0.35,
            )
            st.plotly_chart(fig_devices_status, use_container_width=True)

        with col2:
            shown_columns = [
                col for col in [
                    "device_id",
                    "status",
                    "last_seen_at",
                    "current_model_run_id",
                    "software_version",
                    "camera_name",
                    "location_name",
                    "city",
                ]
                if col in df_devices.columns
            ]

            st.dataframe(
                df_devices[shown_columns],
                use_container_width=True,
            )


def page_alerts() -> None:
    st.title("Alertas")

    col1, col2 = st.columns([1, 1])

    with col1:
        status = st.selectbox(
            "Filtrar por estado",
            ["Todas", "open", "acknowledged", "resolved", "discarded", "closed"],
        )

    with col2:
        limit = st.number_input("Límite", min_value=1, max_value=200, value=50)

    params = {"limit": int(limit)}
    if status != "Todas":
        params["status"] = status

    data = api_request("GET", "/dashboard/alerts", params=params)
    alerts = data.get("items", [])

    df_alerts = to_dataframe(alerts)
    st.dataframe(df_alerts, use_container_width=True)

    st.divider()
    st.subheader("Detalle de alerta")

    alert_ids = [
        item.get("alert_id")
        for item in alerts
        if item.get("alert_id")
    ]

    if not alert_ids:
        st.info("No hay alertas para mostrar.")
        return

    selected_alert_id = st.selectbox("Selecciona una alerta", alert_ids)

    detail = api_request("GET", f"/alerts/{selected_alert_id}")
    alert_item = detail.get("item", {})

    st.json(alert_item)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Evidencia")
        try:
            evidence = api_request("GET", f"/alerts/{selected_alert_id}/evidence-url")
            download_url = evidence.get("download_url")

            if download_url:
                st.image(download_url, caption="Crop de matrícula sospechosa")
                st.caption(evidence.get("s3_key", ""))
            else:
                st.warning("La alerta no tiene URL de evidencia.")
        except Exception as exc:
            st.warning(f"No se pudo cargar evidencia: {exc}")

    with col2:
        st.subheader("Actualizar estado")

        new_status = st.selectbox(
            "Nuevo estado",
            ["open", "acknowledged", "resolved", "discarded", "closed"],
            key="new_alert_status",
        )

        review_notes = st.text_area("Notas de revisión")

        if st.button("Actualizar alerta"):
            payload = {
                "status": new_status,
                "reviewed_by": "streamlit-admin",
                "review_notes": review_notes,
            }

            result = api_request(
                "PATCH",
                f"/alerts/{selected_alert_id}",
                payload=payload,
            )

            st.success("Alerta actualizada")
            st.json(result)
            st.cache_data.clear()


def page_watchlist() -> None:
    st.title("Watchlist")

    data = api_request("GET", "/watchlist")
    items = data.get("items", [])

    st.subheader("Matrículas registradas")
    df = to_dataframe(items)
    st.dataframe(df, use_container_width=True)

    st.divider()
    st.subheader("Añadir matrícula")

    with st.form("add_watchlist_form"):
        plate_text = st.text_input("Matrícula")
        severity = st.selectbox("Severidad", ["low", "medium", "high"])
        reason = st.text_input("Motivo")
        notes = st.text_area("Notas")
        submitted = st.form_submit_button("Añadir")

    if submitted:
        payload = {
            "plate_text": plate_text,
            "status": "active",
            "severity": severity,
            "reason": reason,
            "notes": notes,
            "created_by": "streamlit-admin",
        }

        result = api_request("POST", "/watchlist", payload=payload)
        st.success("Matrícula añadida")
        st.json(result)
        st.cache_data.clear()

    st.divider()
    st.subheader("Actualizar matrícula")

    plate_options = [
        item.get("plate_text")
        for item in items
        if item.get("plate_text")
    ]

    if not plate_options:
        st.info("No hay matrículas para actualizar.")
        return

    selected_plate = st.selectbox("Selecciona matrícula", plate_options)
    new_status = st.selectbox("Estado", ["active", "inactive"])
    new_severity = st.selectbox("Nueva severidad", ["low", "medium", "high"])
    new_reason = st.text_input("Nuevo motivo")
    updated_by = st.text_input("Actualizado por", value="streamlit-admin")

    if st.button("Actualizar matrícula"):
        payload = {
            "status": new_status,
            "severity": new_severity,
            "reason": new_reason,
            "updated_by": updated_by,
        }

        result = api_request(
            "PATCH",
            f"/watchlist/{selected_plate}",
            payload=payload,
        )

        st.success("Watchlist actualizada")
        st.json(result)
        st.cache_data.clear()


def page_devices() -> None:
    st.title("Dispositivos")

    status = st.selectbox(
        "Filtrar por estado",
        ["Todos", "online", "active", "offline", "inactive"],
    )

    params = {}
    if status != "Todos":
        params["status"] = status

    data = api_request("GET", "/dashboard/devices", params=params)
    devices = data.get("items", [])

    df = to_dataframe(devices)
    st.dataframe(df, use_container_width=True)

    if not devices:
        st.info("No hay dispositivos registrados todavía.")
        return

    st.subheader("Última conexión")

    for device in devices:
        device_id = device.get("device_id", "unknown")
        last_seen = device.get("last_seen_at", "sin datos")
        status = device.get("status", "unknown")
        model = device.get("current_model_run_id") or device.get("model_run_id")

        st.write(
            f"**{device_id}** — estado: `{status}` — "
            f"última conexión: `{last_seen}` — modelo: `{model}`"
        )


def main() -> None:
    st.sidebar.title("TFM Dashboard")

    if st.sidebar.button("Recargar datos"):
        st.cache_data.clear()
        st.rerun()

    page = st.sidebar.radio(
        "Secciones",
        [
            "Resumen",
            "Visualización",
            "Alertas",
            "Watchlist",
            "Dispositivos",
        ],
    )

    if page == "Resumen":
        page_summary()
    elif page == "Visualización":
        page_visualizations()
    elif page == "Alertas":
        page_alerts()
    elif page == "Watchlist":
        page_watchlist()
    elif page == "Dispositivos":
        page_devices()


if __name__ == "__main__":
    main()
