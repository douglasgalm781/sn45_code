# Getting Code Completion in VSCODE

## Installing the extension

1. Open VS Code.
2. Press `Ctrl+Shift+X` to open the Extensions view.
3. Search for `Continue.dev` and install it.
4. Restart VS Code.

## Configuring the extension

1. Run the keybinding `Ctrl+Shift+P` to open the Command Palette.
2. Type `Continue.dev: Open config.json` and press `Enter`.
3. This will open the `config.json` file in your workspace.

Now add the following configuration:
# TODO FINISH the below
```json
"models": [
    {
      "title": "Code",
      "model": "code",
      "contextLength": 8000,
      "provider": "openai",
      "apiKey": "EMPTY",
      "apiBase": "http://0.0.0.0:8000/v1"

    }
  ],
"tabAutocompleteModel": {
    "title": "Code",
    "model": "code",
    "contextLength": 8000,
    "provider": "openai",
    "apiKey": "EMPTY",
    "apiBase": "http://0.0.0.0:8000/v1"
  },
```
