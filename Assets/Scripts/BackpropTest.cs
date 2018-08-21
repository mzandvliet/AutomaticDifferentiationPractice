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

    

    Notes:

    - Formulating the correct job order by graph traversal and dependency checking
    for both forward and backward passes only has to be done once.

    - We need to store gradient values for all the parameters
    - Convergence of backprop for any computational graph + config == a mandelbrot set
*/

namespace BackPropPractice {
    public class BackpropTest : MonoBehaviour {

        private void Awake() {
            var a = new ConstNode(5f, true);
            var b = new ConstNode(3f, true);
            var c = new ConstNode(-2f, true);
            
            var add = new AddNode(a, b);
            var mul = new MultiplyNode(c, add);

            var target = new ConstNode(15f, false);
            var loss = new LossNode(mul, target);

            Optimize(loss);
        }

        private static void Optimize(IFloatNode node) {
            const float rate = 0.01f;

            for (int i = 0; i < 10; i++) {
                ForwardPass(node);
                BackwardPass(node);
                ParameterUpdate(node, rate);
            }
        }

        private static void ForwardPass(IFloatNode node) {
            // Create ordered lists of ops, burst-ready

            Debug.Log("Forward pas: " + node);

            var stack = new Stack<IFloatNode>();
            stack.Push(node);

            var list = new List<IFloatNode>();
            
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
            Debug.Log("Backwards pass: " + node);

            var stack = new Stack<IFloatNode>();
            stack.Push(node);

            while (stack.Count > 0) {
                node = stack.Pop();

                float gradient = node.Output.IsConnected ? node.Output.Node.Gradients[node.Output.Inlet] : 1f;
                node.Backward(gradient);

                for (int i = 0; i < node.Inputs.Count; i++) {
                    stack.Push(node.Inputs[i]);
                }
            }
        }

        private static void ParameterUpdate(IFloatNode node, float rate) {
            Debug.Log("Parameter Update");

            var stack = new Stack<IFloatNode>();
            stack.Push(node);

            while (stack.Count > 0) {
                node = stack.Pop();

                if (node is ConstNode) {
                    var cNode = node as ConstNode;
                    if (cNode.IsLearnable) {
                        cNode.Value += node.Output.Node.Gradients[node.Output.Inlet] * rate;
                    }
                }

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

        float Value {
            get;
        }

        float[] Gradients {
            get;
        }

        int Id {
            get;
        }
    }

    public class Outlet {
        public IFloatNode Node {
            get;
            private set;
        }

        public int Inlet {
            get;
            private set;
        }

        public bool IsConnected {
            get { return Node != null; }
        }

        public void Connect(IFloatNode node, int inlet) {
            Node = node;
            Inlet = inlet;
        }
    }

    public abstract class AbstractFloatNode : IFloatNode {
        public IList<IFloatNode> Inputs {
            get;
            private set;
        }

        public Outlet Output {
            get;
        }

        public float Value {
            get;
            set;
        }

        public float[] Gradients {
            get;
            private set;
        }

        public int Id {
            get;
            private set;
        }

        private static int Count;

        public AbstractFloatNode(int inlets) {
            Gradients = new float[inlets];
            Inputs = new IFloatNode[inlets];
            Output = new Outlet();
            Id = Count++;
        }

        public abstract void Forward();
        public abstract void Backward(float gradient);
    }

    public class ConstNode : AbstractFloatNode {
        public bool IsLearnable {
            get;
            private set;
        }

        public ConstNode(float value, bool learnable) : base(0) {
            Value = value;
            IsLearnable = learnable;
        }

        public override void Forward() { }
        public override void Backward(float gradient) {
        }

        public override string ToString() {
            return "ID: " + Id + ", Const: " + Value;
        }
    }

    public class AddNode : AbstractFloatNode {
        public AddNode(IFloatNode a, IFloatNode b) : base(2) {
            Inputs[0] = a;
            Inputs[1] = b;

            a.Output.Connect(this, 0);
            b.Output.Connect(this, 1);
        }

        public override void Forward() {
            Value = Inputs[0].Value + Inputs[1].Value;
        }

        public override void Backward(float gradient) {
            Gradients[0] = gradient;
            Gradients[1] = gradient;
        }

        public override string ToString() {
            return "ID: " + Id + ", + " + Value;
        }
    }

    public class MultiplyNode : AbstractFloatNode {
        public MultiplyNode(IFloatNode a, IFloatNode b) : base(2) {
            Inputs[0] = a;
            Inputs[1] = b;

            a.Output.Connect(this, 0);
            b.Output.Connect(this, 1);
        }

        public override void Forward() {
            Value = Inputs[0].Value * Inputs[1].Value;
        }

        public override void Backward(float gradient) {
            Gradients[0] = gradient * Inputs[1].Value;
            Gradients[1] = gradient * Inputs[0].Value;
        }

        public override string ToString() {
            return "ID: " + Id + ", * " + Value;
        }
    }

    public class LossNode : AbstractFloatNode {
        public LossNode(IFloatNode result, IFloatNode target) : base(2) {
            Inputs[0] = result;
            Inputs[1] = target;

            result.Output.Connect(this, 0);
            target.Output.Connect(this, 1);
        }

        public override void Forward() {
            Value = Inputs[1].Value - Inputs[0].Value;
        }

        public override void Backward(float gradient) {
            // Note: this starts the gradient, since it is always the last node
            Gradients[0] = Value;
            Gradients[1] = Value;
        }

        public override string ToString() {
            return "ID: " + Id + ", Loss: " + Value;
        }
    }

    [AttributeUsage(
    AttributeTargets.Class |
    AttributeTargets.Field |
    AttributeTargets.Property)]
    public class OptimizeParameterAttribute : Attribute {

    }
}
