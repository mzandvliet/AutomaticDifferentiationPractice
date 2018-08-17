using UnityEngine;

/* 
    Let's make some nodes. 

    they're all some f(a, b) thing with a forward
    calculation path, and given a gradient can
    route it back through, with respect to any
    of the inputs.

    We build a graph of these nodes, then execute
    operations on them.

    Todo:
    - Implement the local gradient backprop in the modules themselves

    Instead of in this for loop we have now.
    Important! While tempting to implement forward/backward passes through OO, don't.
    Note: don't calculate all possible derivatives, just the ones you need.

    - Build a graph traversal algorithm that does forward/backward passes automatically
    - Build a state visualizer
    - Implement vector algebra
    - Implement other operations needed by neural nets
    - Implement using Burst

    - Convergence of backprop for any computational graph + config == a mandelbrot set
*/

namespace BackPropPractice {
    public class BackpropTest : MonoBehaviour {

        private void Awake() {
            var a = new ConstNode(5f);
            var b = new ConstNode(3f);
            var add1 = new AddNode(a, b);

            var c = new ConstNode(-2f);
            var add2 = new MultiplyNode(add1, c);

            const float rate = 0.03744485f;

            for (int i = 0; i < 100; i++) {
                float result = add2.Forward();
                float target = 15f;

                float dLdAdd2 = target - result; // Note: get from SumSquareLoss node
                
                float dLdC = add2.BackwardB(dLdAdd2);
                float dLdAdd1 = add2.BackwardA(dLdAdd2);

                float dLdA = add1.BackwardA(dLdAdd1);
                float dLdB = add1.BackwardB(dLdAdd1);

                Debug.Log(i + ": (" + a.Value + " + " + b.Value + ") + " + c.Value + " = " + result);

                a.Value += dLdA * rate;
                b.Value += dLdB * rate;
                c.Value += dLdC * rate;
            }
        }
    }

    public interface IFloatNode {
        float Value {
            get;
        }

        float Forward();
    }

    public class ConstNode : IFloatNode {
        public float Value {
            get;
            set;
        }

        public ConstNode(float value) {
            Value = value;
        }

        public float Forward() {
            return Value;
        }
    }

    public class AddNode : IFloatNode {
        public float Value {
            get;
            private set;
        }

        public IFloatNode A {
            get;
            private set;
        }

        public IFloatNode B {
            get;
            private set;
        }

        public AddNode(IFloatNode a, IFloatNode b) {
            A = a;
            B = b;
        }

        public float Forward() {
            Value = A.Forward() + B.Forward();
            return Value;
        }

        public float BackwardA(float grad) {
            return grad * 1f;
        }

        public float BackwardB(float grad) {
            return grad * 1f;
        }
    }

    public class MultiplyNode : IFloatNode {
        public float Value {
            get;
            private set;
        }

        public IFloatNode A {
            get;
            private set;
        }

        public IFloatNode B {
            get;
            private set;
        }

        public MultiplyNode(IFloatNode a, IFloatNode b) {
            A = a;
            B = b;
        }

        public float Forward() {
            Value = A.Forward() * B.Forward();
            return Value;
        }

        public float BackwardA(float grad) {
            return grad * B.Value;
        }

        public float BackwardB(float grad) {
            return grad * A.Value;
        }
    }
}
