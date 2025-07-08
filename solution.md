## 2.1

### Understanding Unicode

```python
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> print(chr(0).__repr__())
'\x00'
>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```

## 2.2

a)

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # Get the byte values for the encoded string (integers from 0 to 255).
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129,
161, 227, 129, 175, 33]
>>> # One byte does not necessarily correspond to one Unicode character!
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

a)

With UTF-8, common characters (e.g. individual English characters) are encoded using just 1 byte. Rarer characters (e.g. Japanese characters) are encoded using more bytes.

```python
>>> print(list("e".encode("utf-8")))
[101]
>>> print(list("é".encode("utf-8")))
[195, 169]
>>> print(list("こ".encode("utf-8")))
[227, 129, 147]
```

Therefore, most of our data should be nice and compressed.

Also, if we needed to represent all characters in the UTF-32 encoding, we'd start from a very large vocabulary size (154,997), and most of these characters would be extremely rare. We ideally don't want to waste a vocabulary item on very rare characters. By using UTF-8, we can use a smaller base vocabulary and still represent all possible characters.

b)

The function incorrectly assumes that each byte corresponds one-to-one with a character.

For example, the function would fail on the input `café`, since `é` requires 2 bytes to encode.

```python
>>> decode_utf8_bytes_to_str_wrong("café".encode("utf-8"))              
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
  File "<stdin>", line 2, in <listcomp>
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 0: unexpected end of data
```

c)

```python
>>> bytes([128, 128]).decode("utf-8")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

UTF-8 encodes one byte sequences with `0xxxxxxx` (binary representation), and two byte sequences with `110xxxxx 10xxxxxx`. Since 128 corresponds to `10000000`, it cannot be a one byte sequence, nor the start of a two-byte sequence.

## 2.5

BPE Training on TinyStories

a)

TODO


BPE Training on OpenWebText

a)

TODO

b) 

TODO


