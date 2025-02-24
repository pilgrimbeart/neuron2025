# neuron2025
We can implement simple "gate-level" functionality:

Wire
And
Or
Xor (?)
Inhibit (which is as close to 'not' as pulse-based systems can come)
Oscillator

We know that in static logic, you can build a universal computer out of "NAND plus register"
A good test component for pulse logic is the adder:

In static logic, this consists of two components:
    . For bit0 out, an XOR gate
    . For carry out, an AND gate

So can we implement an XOR gate?
