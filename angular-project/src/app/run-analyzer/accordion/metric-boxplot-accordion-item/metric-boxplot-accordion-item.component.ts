import {Component, Input, OnInit, ElementRef, AfterViewInit, OnChanges} from "@angular/core";
import {Chart, LinearScale, CategoryScale} from 'chart.js';
import {BoxPlotController, BoxAndWiskers} from '@sgratzl/chartjs-chart-boxplot';
import html2canvas from "html2canvas";
import {mod} from "ngx-bootstrap/chronos/utils";
// register controller in chart.js and ensure the defaults are set
Chart.register(BoxPlotController, BoxAndWiskers, LinearScale, CategoryScale);


function calculateQ1(numbers: number[]): number {
  const sortedNumbers = numbers.slice().sort((a, b) => a - b);
  const n = sortedNumbers.length;
  const index = (n + 1) / 4;
  if (Number.isInteger(index)) {
    return sortedNumbers[index - 1];
  } else {
    const lowerIndex = Math.floor(index);
    const upperIndex = Math.ceil(index);
    return (sortedNumbers[lowerIndex - 1] + sortedNumbers[upperIndex - 1]) / 2;
  }
}

function calculateQ3(numbers: number[]): number {
  const sortedNumbers = numbers.slice().sort((a, b) => a - b);
  const n = sortedNumbers.length;
  const index = (3 * (n + 1)) / 4;
  if (Number.isInteger(index)) {
    return sortedNumbers[index - 1];
  } else {
    const lowerIndex = Math.floor(index);
    const upperIndex = Math.ceil(index);
    return (sortedNumbers[lowerIndex - 1] + sortedNumbers[upperIndex - 1]) / 2;
  }

}

function calculateMean(numbers: number[]): number {
  const sum = numbers.reduce((acc, curr) => acc + curr, 0);
  return sum / numbers.length;
}

function calculateMedian(numbers: number[]): number {
  const sortedNumbers = numbers.slice().sort((a, b) => a - b);
  const n = sortedNumbers.length;
  const middle = Math.floor(n / 2);
  if (n % 2 === 0) {
    return (sortedNumbers[middle - 1] + sortedNumbers[middle]) / 2;
  } else {
    return sortedNumbers[middle];
  }
}

function min(numbers: number[]) {
  let min = numbers[0]
  for (let num of numbers) {
    if (num < min) {
      min = num;
    }
  }
  return min;
}

function max(numbers: number[]) {
  let max = numbers[0]
  for (let num of numbers) {
    if (num > max) {
      max = num;
    }
  }
  return max;
}

@Component({
  selector: 'app-metric-boxplot-accordion-item',
  templateUrl: './metric-boxplot-accordion-item.component.html',
  styleUrls: ['./metric-boxplot-accordion-item.component.scss']
})

export class MetricBoxplotAccordionItemComponent implements OnInit, OnChanges {
  @Input() metrics: string[] = [];
  @Input() totalMetricData: any | null = null;
  @Input() sensitive: any = {};
  @Input() allRuns: string[] | null = null;
  allModels: string[] = []
  metric: string = "error_rate";
  runNames: string[] = []
  runName: string = "";
  myChart: Chart | null = null;




  constructor(private readonly elementRef: ElementRef) {
  }

  ngOnInit() {
    this.runNames = Object.keys(this.totalMetricData[1])
    this.runName = this.runNames[0];
    this.allModels = []
    // go over every run and add every different model
    if (this.allRuns) {
      for (let run of this.allRuns) {
        for (let model in this.totalMetricData[0][run]) {
          let shortenModelName = model
          if(shortenModelName.endsWith("_tuned")){
            shortenModelName = shortenModelName.replace("_tuned", "")
          }
          if (!this.allModels.includes(shortenModelName)) {
            this.allModels.push(shortenModelName)
          }
        }
      }
    }
    this.createBoxPlot();
  }

  ngOnChanges() {
    this.runNames = Object.keys(this.totalMetricData[1])
    this.runName = this.runNames[0];
    this.allModels = []
    // go over every run and add every different model
    if (this.allRuns) {
      for (let run of this.allRuns) {
        for (let model in this.totalMetricData[0][run]) {
          let shortenModelName = model
          if(shortenModelName.endsWith("_tuned")){
            shortenModelName = shortenModelName.replace("_tuned", "")
          }
          if (!this.allModels.includes(shortenModelName)) {
            this.allModels.push(shortenModelName)
          }
        }
      }
    }
    this.createBoxPlot();
  }


  setMetricDataPoints() {
    if (this.myChart && this.totalMetricData[1] != null) {
      let datasets = []
      for(let run of this.runNames){
        let multipleData = []
        for(let model of this.allModels){
          let data = []
          // check if this run trained this model
          let trained = false
          for(let run_model in this.totalMetricData[0][run]){
            if(run_model == model || run_model == model + "_tuned"){
              trained = true
            }
          }
          if(trained){
            for(const iterationName in this.totalMetricData[1][run]){
              if(iterationName.startsWith(model)){
                data.push(this.totalMetricData[1][run][iterationName][this.metric])
              }
            }
          }
          multipleData.push(data)
        }
        datasets.push({
        label: run,
        data: multipleData,
        borderWidth: 2,
        itemRadius: 4,
        itemStyle: 'circle',
        itemBackgroundColor: "#000",
        outlierBackgroundColor: '#000',
        })
      }
      console.log("---------")
      console.log(this.totalMetricData[1]["RF"])
      console.log(this.allModels)
      this.myChart.config.data = {labels: this.allModels, datasets: datasets};
      this.myChart.update();

    }

  }


  showBoxPlot() {
    const configMetricChart: any = {
      type: 'boxplot',
      data: {},
      options: {
        scales: {
          y: {
            beginAtZero: false,
            title: {
              display: true,
              text: ""
            }
          }
        }
      },
    };
    // Create the scatter charts

    if (this.myChart) {
      this.myChart.destroy();
    }
    this.myChart = new Chart('analyzer-boxplot-chart', configMetricChart);
  }


  createBoxPlot() {
    this.showBoxPlot();
    this.setMetricDataPoints();
  }

  combinedMetric = ["error_rate"]
  weights = [1]
  metricName = ""

  addMetric() {
    this.combinedMetric.push("");
    this.weights.push();
  }

  openMetricCreaterModal() {
    const modelDiv = document.getElementById("metricCreaterModalBox")
    if (modelDiv != null) {
      modelDiv.style.display = 'block';
    }
  }

  closeMetricCombinerModel() {
    const modelDiv = document.getElementById("metricCreaterModalBox")
    if (modelDiv != null) {
      modelDiv.style.display = 'none';
    }
  }

  removeMetric(index: number) {
    if (this.combinedMetric.length > 1) {
      this.combinedMetric.splice(index, 1)
      this.weights.splice(index, 1)
    }
  }

  addNewMetric() {
    let totalWeight = 0
    for (let index = 0; index < this.weights.length; index++) {
      totalWeight = totalWeight + Number(this.weights[index]);
    }
    this.metrics.push(this.metricName)
    for (let run in this.totalMetricData[0]) {
      for (let subRun in this.totalMetricData[0][run]) {
        let totalValue = 0;
        for (let metric in this.totalMetricData[0][run][subRun]) {
          let index = this.combinedMetric.indexOf(metric);
          if (index !== -1) {
            let weight = this.weights[index]
            let value = this.totalMetricData[0][run][subRun][metric]
            totalValue = totalValue + (weight / totalWeight) * value;
          }
        }
        this.totalMetricData[0][run][subRun][this.metricName] = totalValue
      }
    }
    for (let run in this.totalMetricData[1]) {
      for (let subRun in this.totalMetricData[1][run]) {
        let totalValue = 0;
        for (let metric in this.totalMetricData[1][run][subRun]) {
          let index = this.combinedMetric.indexOf(metric);
          if (index !== -1) {
            let weight = this.weights[index]
            let value = this.totalMetricData[1][run][subRun][metric]

            totalValue = totalValue + (weight / totalWeight) * value;

          }
        }
        this.totalMetricData[1][run][subRun][this.metricName] = totalValue
      }
    }
    this.closeMetricCombinerModel();
  }

  takeScreenshot() {
    // @ts-ignore
    html2canvas(document.getElementById("take-screenshot"), {logging: false})
      .then(canvas => this.startDownload("BoxPlot.png", canvas.toDataURL("image/png;base64")))
  }


  filterRunName(name: string) {
    return name.split("-")[0]
  }

  downloadCSV() {
    console.log(this.myChart)
    let csvString = "model , min, 1.quantile, median, mean, 2.quantile, max \n"
    for (let model of this.runNames) {
      let data = []
      for (const modelName in this.totalMetricData[1][model]) {
        data.push(this.totalMetricData[1][model][modelName][this.metric])
      }
      csvString = csvString + model + ", " + min(data) + ", " + calculateQ1(data) + ", " + calculateMedian(data) + ", " + calculateMean(data) + ", " +
        calculateQ3(data) + ", " + max(data) + "\n";
    }
    const blob = new Blob([csvString], {type: 'text/csv;charset=utf-8;'});
    this.startDownload("BoxPlot", URL.createObjectURL(blob));
  }

  // @ts-ignore
  startDownload(name, href) {
    let htmlElement = document.createElement("a");
    htmlElement.href = href;
    htmlElement.download = name;
    htmlElement.click();
  }


}
