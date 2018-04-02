using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;
using Newtonsoft.Json;

namespace CNTKMnist
{
	class Program
	{
		static void Main(string[] args)
		{
			Random rng = new Random(0);

			int batch_size = 128;
			int num_classes = 10;
			int epochs = 120;

			var device = DeviceDescriptor.GPUDevice(0);
			var modelFunc = Function.Load("../../../../keras/model.dnn", device);

			var labelsVar = CNTKLib.InputVariable(new int[] { modelFunc.Output.Shape.TotalSize }, DataType.Float, new AxisVector() { modelFunc.Output.DynamicAxes[0] });

			//TODO: This is probably wrong?
			// https://github.com/keras-team/keras/blob/master/keras/losses.py#L68
			// https://github.com/keras-team/keras/blob/master/keras/backend/cntk_backend.py#L1745
			var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(modelFunc, labelsVar, "lossFunction");
			
			//Keras calls this metrics='accuracy', not sure if correct
			//CNTKLib.ReduceSum()
			var prediction = CNTKLib.ClassificationError(modelFunc, labelsVar, "predictionError");

			var trainer = Trainer.CreateTrainer(modelFunc, trainingLoss, prediction, new List<Learner>
			{
				Learner.SGDLearner(modelFunc.Parameters(), new TrainingParameterScheduleDouble(0.01, (uint)batch_size))

				//TODO: Is this the right way to make a ParameterVector?
				//CNTKLib.AdaDeltaLearner(new ParameterVector(new List<Parameter>(modelFunc.Parameters())), new TrainingParameterScheduleDouble(1))
			});



			//TODO: Load data
			//x is input 2d image, y is output class
			// _train is training data, _test is evaluation data
			Console.WriteLine("Loading data sets");
			var x_train = JsonConvert.DeserializeObject<int[][][]>(File.ReadAllText("../../../../keras/x_train.json"));
			var y_train = JsonConvert.DeserializeObject<int[]>(File.ReadAllText("../../../../keras/y_train.json"));

			var x_test = JsonConvert.DeserializeObject<int[][][]>(File.ReadAllText("../../../../keras/x_test.json"));
			var y_test = JsonConvert.DeserializeObject<int[]>(File.ReadAllText("../../../../keras/y_test.json"));



			for (var e = 0; e < epochs; e++)
			{
				Console.WriteLine("Epoch " + e);

				//TODO: Evaluate
				{
					Console.WriteLine("Performing evaluation");
					int correct = 0;

					//Copy data to vals
					float[] vals = new float[28 * 28 * 10000];
					int v = 0;
					for (var i = 0; i < x_test.Length; i++)
					{
						for (var y = 0; y < 28; y++)
						for (var x = 0; x < 28; x++)
							vals[v++] = x_test[i][x][y] / 255f;
					}

					var inputVar = modelFunc.Arguments.Single();
					var outputVar = modelFunc.Output;

					var inputDataMap = new Dictionary<Variable, Value>();
					var inputVal = Value.CreateBatch(inputVar.Shape, vals, device);
					inputDataMap.Add(inputVar, inputVal);

					var outputDataMap = new Dictionary<Variable, Value>();
					outputDataMap.Add(outputVar, null);

					modelFunc.Evaluate(inputDataMap, outputDataMap, device);

					var outputVal = outputDataMap[outputVar];
					var outputData = outputVal.GetDenseData<float>(outputVar);

					for (var b = 0; b < x_test.Length; b++)
					{
						var ourData = outputData[b];

						int bestIndex = 0;
						float bestScore = ourData[0];
						for (var i = 1; i < 10; i++)
						{
							if (ourData[i] > bestScore)
							{
								bestIndex = i;
								bestScore = ourData[i];
							}
						}

						if (bestIndex == y_test[b])
							correct++;
					}

					//https://github.com/Microsoft/CNTK/issues/2954
					inputVal.Erase();
					outputVal.Erase();

					Console.WriteLine($"Correct {correct}");
				}


				//Train
				{
					Console.WriteLine("Performing training");
					var input = modelFunc.Arguments.Single();

					//Shuffle x_train/y_train
					for (var i = 0; i < x_train.Length; i++)
					{
						int k = rng.Next(i, x_train.Length);

						var value = x_train[k];
						x_train[k] = x_train[i];
						x_train[i] = value;

						var value2 = y_train[k];
						y_train[k] = y_train[i];
						y_train[i] = value2;
					}

					//In batches of batch_size, run training
					for (var batch = 0; batch < x_train.Length / batch_size; batch++)
					{
						var trainingData = new float[batch_size * 28 * 28];
						var labelData = new float[batch_size * 10];
						var v = 0;

						for (var i = 0; i < batch_size; i++)
						{
							for (var y = 0; y < 28; y++)
							for (var x = 0; x < 28; x++)
								trainingData[v++] = x_train[i + (batch_size * batch)][x][y] / 255f;

							//1 hot encoding of the label
							labelData[i * 10 + y_train[i + (batch_size * batch)]] = 1;
						}


						var inputBatch = Value.CreateBatch(input.Shape, trainingData, 0, batch_size * 28 * 28, device);
						var labelBatch = Value.CreateBatch(modelFunc.Output.Shape, labelData, 0, batch_size * 10, device);

						var arguments = new Dictionary<Variable, Value>
						{
							{ input, inputBatch },
							{ labelsVar, labelBatch }
						};

						trainer.TrainMinibatch(arguments, false, device);

						//https://github.com/Microsoft/CNTK/issues/2954
						inputBatch.Erase();
						labelBatch.Erase();
					}
				}
			}
		}
	}
}
