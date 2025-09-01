# Guía de contribución

Gracias por tu interés en contribuir a **TNFR**.

## Versión del proyecto

La versión oficial del proyecto se define **solo** en [`pyproject.toml`](./pyproject.toml) dentro de la sección `[project]`.

- `package.json` no incluye un campo `version`; cualquier build de Node debe consultar `pyproject.toml` si necesita el número de versión.
- No agregues valores de versión en otros archivos.

## Flujo de trabajo

1. Crea un *fork* y una rama para tu cambio.
2. Instala las dependencias de Python y ejecuta las pruebas:
   ```bash
   pip install -e .[dev]
   pytest
   ```
3. Asegúrate de que tus cambios sigan la estructura del proyecto y estén cubiertos por pruebas cuando sea posible.
4. Abre un *pull request* describiendo brevemente tu contribución.

¡Gracias por ayudar a mejorar TNFR!
