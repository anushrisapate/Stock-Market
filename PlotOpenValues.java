import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PlotOpenValues extends JFrame {

    public PlotOpenValues(double[] allValues, double[] predictedValues) {
        // Create a series for all the actual values in the dataset
        XYSeries allSeries = new XYSeries("All Open Values");
        for (int i = 0; i < allValues.length; i++) {
            allSeries.add(i, allValues[i]);
        }

        // Create a series for the predicted values starting from index 620 and only plotting 10 values
        XYSeries predictedSeries = new XYSeries("Predicted Open Values");
        int predictedIndex = 620; // Starting index for predicted values
        int count = 0; // Counter for plotted values

        for (int i = 0; i < predictedValues.length; i++) {
            // Only plot if the value is greater than 130 and if we haven't plotted 10 values yet
            if (predictedValues[i] > 130 && count < 10) {
                predictedSeries.add(predictedIndex + i, predictedValues[i]); // Plot corresponding predicted value
                count++;
            }
            // Stop if we have plotted 10 values
            if (count >= 10) break;
        }

        // Add the series to a dataset
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(allSeries);
        dataset.addSeries(predictedSeries);

        // Create the chart
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Stock Open Values - Actual vs Predicted", // Chart title
                "Index",                                    // X-Axis Label
                "Open Value",                               // Y-Axis Label
                dataset,                                    // Dataset
                PlotOrientation.VERTICAL,
                true,                                       // Show legend
                true,
                false
        );

        // Create a panel for the chart
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(800, 600));
        setContentPane(chartPanel);
    }

    public static void main(String[] args) {
        // Load the dataset
        String filePath = "/Users/shivamsanap/Desktop/DAACP/zomato_stock_price.csv"; // Update this with the actual file path
        double[] openValues = loadDataset(filePath);

        // Define the range for the training dataset (index 0 to 600)
        int startTrainingIndex = 0;
        int trainingDataSize = 600;

        // Divide dataset into training and prediction sets
        double[] trainingData = new double[trainingDataSize];
        System.arraycopy(openValues, startTrainingIndex, trainingData, 0, trainingDataSize);

        // Use the remaining data points for actual values (30 data points)
        int remainingDataSize = 30;
        double[] actualValues = new double[remainingDataSize];
        System.arraycopy(openValues, trainingDataSize, actualValues, 0, remainingDataSize);

        // PSO Initialization and execution
        double[] predictedValues = performPSO(trainingData, remainingDataSize); // Change to remainingDataSize

        // Create the chart window
        SwingUtilities.invokeLater(() -> {
            PlotOpenValues example = new PlotOpenValues(openValues, predictedValues);
            example.setSize(800, 600);
            example.setLocationRelativeTo(null);
            example.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            example.setVisible(true);
        });
    }

    // Perform PSO and return predicted values
    static double[] performPSO(double[] trainingData, int predictionLength) {
        Particle[] particles = initializeParticles(trainingData);
        globalBestPosition = new double[trainingData.length];

        // PSO Iterations
        for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
            for (Particle p : particles) {
                double fitness = calculateFitness(p.position, trainingData);

                if (fitness < p.bestFitness) {
                    p.bestFitness = fitness;
                    p.bestPosition = p.position.clone();
                }

                // Update global best
                if (fitness < globalBestFitness) {
                    globalBestFitness = fitness;
                    globalBestPosition = p.position.clone();
                }
            }

            // Update particle positions and velocities
            for (Particle p : particles) {
                updateParticle(p);
            }
        }

        // Predict future values using the global best position
        return predictNextValues(globalBestPosition, predictionLength);
    }

    // Function to load the dataset
    static double[] loadDataset(String filePath) {
        List<Double> openValues = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            br.readLine(); // Skip the header
            while ((line = br.readLine()) != null) {
                String[] data = line.split(",");
                openValues.add(Double.parseDouble(data[1])); // Assuming 'Open' is the second column
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return openValues.stream().mapToDouble(Double::doubleValue).toArray();
    }

    // Initialize particles randomly
    static Particle[] initializeParticles(double[] trainingData) {
        Particle[] particles = new Particle[NUM_PARTICLES];
        Random rand = new Random();
        for (int i = 0; i < NUM_PARTICLES; i++) {
            double[] position = new double[trainingData.length];
            double[] velocity = new double[trainingData.length];
            for (int j = 0; j < trainingData.length; j++) {
                position[j] = trainingData[j] + rand.nextDouble() * 0.1 - 0.05; // Randomly adjust within a small range
                velocity[j] = rand.nextDouble() * 0.2 - 0.1;
            }
            particles[i] = new Particle(position, velocity);
        }
        return particles;
    }

    // Fitness function: Mean Squared Error (MSE)
    static double calculateFitness(double[] position, double[] trainingData) {
        double mse = 0.0;
        for (int i = 0; i < trainingData.length; i++) {
            mse += Math.pow(position[i] - trainingData[i], 2);
        }
        return mse / trainingData.length;
    }

    // Update particle's velocity and position
    static void updateParticle(Particle particle) {
        Random rand = new Random();
        for (int i = 0; i < particle.position.length; i++) {
            double r1 = rand.nextDouble();
            double r2 = rand.nextDouble();
            particle.velocity[i] = INERTIA_WEIGHT * particle.velocity[i] +
                    COGNITIVE_COMPONENT * r1 * (particle.bestPosition[i] - particle.position[i]) +
                    SOCIAL_COMPONENT * r2 * (globalBestPosition[i] - particle.position[i]);
            particle.position[i] += particle.velocity[i];
        }
    }

    // Predict future values using the global best position
    static double[] predictNextValues(double[] bestPosition, int length) {
        double[] predictions = new double[length];
        for (int i = 0; i < length; i++) {
            predictions[i] = bestPosition[i % bestPosition.length]; // Repeat the pattern for the prediction length
        }
        return predictions;
    }

    // Particle class definition
    static class Particle {
        double[] position;
        double[] velocity;
        double[] bestPosition;
        double bestFitness = Double.MAX_VALUE;

        public Particle(double[] position, double[] velocity) {
            this.position = position;
            this.velocity = velocity;
            this.bestPosition = position.clone();
        }
    }

    // PSO parameters
    static final int NUM_PARTICLES = 30;
    static final int MAX_ITERATIONS = 100;
    static final double INERTIA_WEIGHT = 0.7;
    static final double COGNITIVE_COMPONENT = 1.5;
    static final double SOCIAL_COMPONENT = 1.5;

    // Global best variables
    static double[] globalBestPosition;
    static double globalBestFitness = Double.MAX_VALUE;
}