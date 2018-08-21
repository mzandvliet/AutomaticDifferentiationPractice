using System;
using System.Collections.Generic;
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

    We need an inlet/outlet system
    Note: don't calculate all possible derivatives, just the ones you need.

    - Build a graph traversal algorithm that does forward/backward passes automatically
    - Build a state visualizer
    - Implement vector algebra
    - Implement other operations needed by neural nets
    - Implement using Burst

    - Convergence of backprop for any computational graph + config == a mandelbrot set


    We need to store gradient values for all the parameters
*/

namespace BackPropPractice {
    public class BackpropTest : MonoBehaviour {

        private void Awake() {
            var a = new ConstNode(5f);
            var b = new ConstNode(3f);
            var c = new ConstNode(-2f);

            IFloatNode node = new AddNode(a, b);
            node = new MultiplyNode(c, node);

            Optimize(node);
        }

        private static void Optimize(IFloatNode node) {
            const float rate = 0.01f;

            for (int i = 0; i < 1; i++) {
                ForwardPass(node);
                float result = node.ForwardValue;
                float target = 15f;

                float dLdO = target - result; // Note: get from SumSquareLoss node

                // Todo: visualize the graph
                //Debug.Log(i + ": " + a.ForwardValue + " + " + b.ForwardValue + " = " + result);

                // a.ForwardValue += dLdA * rate;
                // b.ForwardValue += dLdB * rate;
                // c.ForwardValue += dLdC * rate;
            }
        }

        private static void ForwardPass(IFloatNode node) {
            // Create ordered lists of ops, burst-ready

            Debug.Log("Forward pass");

            var stack = new Stack<IFloatNode>();
            var list = new List<IFloatNode>();

            stack.Push(node);
            
            while (stack.Count > 0) {
                node = stack.Pop();

                list.Add(node);

                for (int i = 0; i < node.Inputs.Count; i++) {
                    stack.Push(node.Inputs[i]);
                }
            }

            list.Reverse();

            for (int i = 0; i < list.Count; i++) {
                list[i].Forward();
                Debug.Log(i + ": " + list[i].ToString());
            }
        }

        private static void BackwardPass(IFloatNode node) {
            var stack = new Stack<IFloatNode>();
            stack.Push(node);

            float gradient = 1f;

            while (stack.Count > 0) {
                var n = stack.Pop();

                node.Backward(gradient);

                for (int i = 0; i < node.Inputs.Count; i++) {
                    stack.Push(node.Inputs[i]);
                }
            }
        }
    }

    public interface IFloatNode {
        IList<IFloatNode> Inputs {
            get;
        }

        Outlet Output {
            get;
        }

        void Forward();
        void Backward(float gradient);

        float ForwardValue {
            get;
        }

        float[] BackwardValues {
            get;
        }

        int Id {
            get;
        }
    }

    public class Outlet {
        IFloatNode Node;
        int Inlet;

        public bool IsConnected {
            get { return Node != null; }
        }

        public void Connect(IFloatNode node, int inlet) {
            Node = node;
            Inlet = inlet;
        }
    }

    public abstract class AbstractNode : IFloatNode {
        public IList<IFloatNode> Inputs {
            get;
            private set;
        }

        public Outlet Output {
            get;
        }

        public float ForwardValue {
            get;
            set;
        }

        public float[] BackwardValues {
            get;
            private set;
        }

        public int Id {
            get;
            private set;
        }

        private static int Count;

        public AbstractNode(int inlets) {
            BackwardValues = new float[inlets];
            Inputs = new IFloatNode[inlets];
            Output = new Outlet();
            Id = Count++;
        }

        public abstract void Forward();
        public abstract void Backward(float gradient);
    }

    public class ConstNode : AbstractNode {
        public ConstNode(float forwardValue) : base(0) {
            ForwardValue = forwardValue;
        }

        public override void Forward() { }
        public override void Backward(float gradient) { }

        public override string ToString() {
            return "ID: " + Id + ", Const: " + ForwardValue;
        }
    }

    public class AddNode : AbstractNode {
        public AddNode(IFloatNode a, IFloatNode b) : base(2) {
            Inputs[0] = a;
            Inputs[1] = b;

            a.Output.Connect(this, 0);
            b.Output.Connect(this, 1);
        }

        public override void Forward() {
            ForwardValue = Inputs[0].ForwardValue + Inputs[1].ForwardValue;
        }

        public override void Backward(float gradient) {
            BackwardValues[0] = gradient;
            BackwardValues[1] = gradient;
        }

        public override string ToString() {
            return "ID: " + Id + ", + " + ForwardValue;
        }
    }

    public class MultiplyNode : AbstractNode {
        public MultiplyNode(IFloatNode a, IFloatNode b) : base(2) {
            Inputs[0] = a;
            Inputs[1] = b;

            a.Output.Connect(this, 0);
            b.Output.Connect(this, 1);
        }

        public override void Forward() {
            ForwardValue = Inputs[0].ForwardValue * Inputs[1].ForwardValue;
        }

        public override void Backward(float gradient) {
            BackwardValues[0] = gradient * Inputs[1].ForwardValue;
            BackwardValues[1] = gradient * Inputs[0].ForwardValue;
        }

        public override string ToString() {
            return "ID: " + Id + ", * " + ForwardValue;
        }
    }

    // public class LossNode : AbstractNode {
    //     public LossNode(IFloatNode result, IFloatNode target) : base(2) {
    //         Inputs[0] = result;
    //         Inputs[1] = target;

    //         result.Output = this;
    //         target.Output = this;
    //     }

    //     public void Forward() {
    //         ForwardValue = Inputs[1].ForwardValue - Inputs[0].ForwardValue;
    //     }

    //     public void Backward(int param) {
    //     }
    // }

    [AttributeUsage(
    AttributeTargets.Class |
    AttributeTargets.Field |
    AttributeTargets.Property)]
    public class OptimizeParameterAttribute : Attribute {

    }
}
