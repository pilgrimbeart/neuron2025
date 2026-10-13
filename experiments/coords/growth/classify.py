import numpy as np
def classify(z):
    """The corners' pattern, round the sheet: 'right' (four distinct corners of the square, in order), 'same axis'
    (u and v both split the corners the same way: two pairs), 'local' (a 3:1 or alternating split: not a smooth
    large-scale pattern), or 'other'."""
    cs = np.sign([z[0, 0], z[0, -1], z[-1, -1], z[-1, 0]])
    def kind(col):
        s = list(col)
        if abs(sum(s)) == 4: return "flat"
        if abs(sum(s)) == 2: return "3:1"
        if s[0] == s[1] and s[2] == s[3]: return "A"          # split between corners 1-2 and 3-4
        if s[1] == s[2] and s[3] == s[0]: return "B"
        return "alt"
    ku, kv = kind(cs[:, 0]), kind(cs[:, 1])
    if {ku, kv} == {"A", "B"}: return "right"
    if ku in "AB" and kv in "AB": return "same axis"
    return "local"
