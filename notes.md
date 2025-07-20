
#### Parameter initialization

```python
t.size()                  # torch.Size object (like a tuple)
t.stride()                # tuple of ints
t.storage().size()        # #elements in underlying buffer
t.element_size()          # bytes per scalar
t.nelement()              # total logical elements
t.is_contiguous(memory_format='contiguous' or 'channels_last')
t.data_ptr()              # int address
```

