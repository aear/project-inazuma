# Local credentials

Keep real tokens outside version control. Discord can resolve its token from
an environment variable or a local file. For a local operator configuration:

```json
{"discord": {"token_file": ".secrets/discord_bot_token"}}
```

The configured environment variable takes precedence over the token file.
`.secrets/discord_bot_token.example` is a template; never put a real token there.
Real `.secrets/*` files remain ignored. Restrict a real token file to its owner.
