# ferrum.config

Process-level configuration store for ferrum defaults.

Uses `contextvars.ContextVar` for thread-safe, scope-bounded defaults.
The `defaults()` context manager provides temporary overrides that revert on exit.

**Precedence (highest wins):**

1. Explicit per-chart `.properties()`
2. `ferrum.config` overrides
3. Built-in defaults

::: ferrum.config
    options:
      members_order: source
      show_root_heading: false
      show_root_toc_entry: false
      filters: ["!^_"]
