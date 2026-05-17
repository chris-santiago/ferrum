# ferrum.legend

Per-channel legend configuration.

The `Legend` dataclass controls legend appearance — title, orientation, direction,
symbols, gradients, and positioning. Pass a `Legend` instance to an encoding
channel's `legend=` parameter, or use `legend=False` to suppress the legend.

::: ferrum.legend.Legend
    options:
      members_order: source
      show_root_heading: true
      show_root_toc_entry: true
