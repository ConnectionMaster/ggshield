### Added

- API tokens are now stored in the OS credential store (macOS Keychain, Windows Credential Locker, Linux Secret Service) via the `keyring` library instead of cleartext in `auth_config.yaml`. Existing cleartext tokens are migrated automatically the next time the configuration is saved. If no OS credential store is available or `GGSHIELD_NO_KEYRING=1`, file-based storage is used as a fall-back.
