import {Component, Input, OnChanges, OnInit} from '@angular/core';
import {Chart, TooltipModel} from "chart.js";
import {mod} from "ngx-bootstrap/chronos/utils";
import html2canvas from "html2canvas";

@Component({
  selector: 'app-metric-barchart-accordion-item',
  templateUrl: './metric-barchart-accordion-item.component.html',
  styleUrls: ['./metric-barchart-accordion-item.component.scss']
})
export class MetricBarchartAccordionItemComponent implements OnInit, OnChanges {
  @Input() metrics: string[] = [];
  @Input() totalMetricData: any | null = null;
  metric: string = "error_rate";

  switchStatus: any;
  myChart: Chart | null = null;
  myRunChart: Chart | null = null;
  showAvg: boolean = true;
  shownRun: string | null = null
  shownModel: string | null = null
  metric1: string = "error_rate";
  metric2: string = "demographic_parity";
  metric3: string = "lrd_dp";
  allModels: string[] = [];
  @Input() allRuns: string[] | null = null;

  ngOnInit() {
    if (this.allRuns) {
      this.shownRun = this.allRuns[0];
      // go over every run and add every different model
      for (let run of this.allRuns) {
        for (let model in this.totalMetricData[0][run]) {
          let shortenModelName = model
          const suffix = "_tuned"
          if (model.endsWith("_tuned")) {
            shortenModelName = shortenModelName.slice(0, -suffix.length)
          }
          if (!this.allModels.includes(shortenModelName)) {
            this.allModels.push(shortenModelName)
          }
        }
      }
      this.shownModel = this.allModels[0]
    }
    this.createMetricBarChart();
    this.createMetricBarChartForRuns();
  }

  ngOnChanges() {
    if (this.allRuns) {
      this.shownRun = this.allRuns[0];
    }
    // this.createMetricBarChart();
    this.createMetricBarChartForRuns();
  }


  filterRunName(name: string) {
    return name.split("-")[0]
  }

  createMetricBarChart() {
    this.showMetricScatter();
    this.setMetricDataPoints2();
  }

  createMetricBarChartForRuns() {
    this.showMetricScatterForRun();
    this.setMetricDataPointForRunChart();
  }


  setMetricDataPointForRunChart() {
    let shownMetrics = [this.metric1, this.metric2, this.metric3];
    if (this.myRunChart && this.totalMetricData[0] != null && this.shownRun) {
      let labels = []
      let datasets = []
      // get  every label
      for (const modelName in this.totalMetricData[0][this.shownRun]) {
        // TODO dictionary mapping to short name
        let shortenName = modelName
        labels.push(this.filterRunName(shortenName))

      }
      // get a dataset for every selected metric
      for (const metric of shownMetrics) {
        let dataset = {}
        // go over every Run and select the values corresponding to the metric
        let data = []
        for (const modelName in this.totalMetricData[0][this.shownRun]) {
          //if(modelName.startsWith("FALCC")){
          data.push(this.totalMetricData[0][this.shownRun][modelName][metric])
          //}
        }
        dataset = {
          label: metric,
          data: data
        }

        datasets.push(dataset)
      }

      console.log(datasets)
      // @ts-ignore
      this.myRunChart.config.data = {labels: labels, datasets: datasets}
      this.myRunChart.update();
    }
  }

  updateConfig() {
    this.setMetricDataPointForRunChart();
    this.setMetricDataPoints2();
  }

  setMetricDataPoints2() {
    if (this.myChart && this.totalMetricData[0] != null && this.shownRun && this.allRuns) {
      let shownMetrics = [this.metric1, this.metric2, this.metric3];
      let datasets = []
      let runLabels = []
      // get the labels:
      for (const runName of this.allRuns) {
        for (const modelName in this.totalMetricData[0][runName]) {
          if (modelName == this.shownModel || modelName == this.shownModel + "_tuned") {
            runLabels.push(runName)
          }
        }
      }
      for (let metric of shownMetrics) {
        let data = []
        // get every run that trained the specific model
        for (const runName of this.allRuns) {
          for (const modelName in this.totalMetricData[0][runName]) {
            if (modelName == this.shownModel || modelName == this.shownModel + "_tuned") {
              data.push(this.totalMetricData[0][runName][modelName][metric])
            }
          }
        }
        datasets.push({label: metric, data: data})
      }
      this.myChart.config.data = {labels: runLabels, datasets: datasets}
      this.myChart.update();
    }
  }


  showMetricScatter() {
    const configMetricChart: any = {
      type: 'bar',
      data: {},
      options: {
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: this.metric
            }
          },
        }
      },
    };
    // Create the scatter charts
    if (this.myChart) {
      this.myChart.destroy();
    }
    this.myChart = new Chart('analyzer-bar-chart', configMetricChart);
  }

  showMetricScatterForRun() {
    const configMetricChart: any = {
      type: 'bar',
      data: {},
      options: {
        scales: {
          y: {
            beginAtZero: true
          },
        }
      },
    };
    // Create the scatter charts
    if (this.myRunChart) {
      this.myRunChart.destroy();
    }
    this.myRunChart = new Chart('analyzer-bar-chart-Run', configMetricChart);
  }


  test() {
    console.log(this.totalMetricData[0])
  }

  combinedMetric = ["accuracy"]
  weights = [1]
  metricName = ""


  addMetric() {
    this.combinedMetric.push("");
    this.weights.push();
  }

  openMetricCreaterModal() {
    const modelDiv = document.getElementById("metricCreaterModalBar")
    if (modelDiv != null) {
      modelDiv.style.display = 'block';
    }
  }

  closeMetricCombinerModel() {
    const modelDiv = document.getElementById("metricCreaterModalBar")
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

  takeScreenshotRun() {
    // @ts-ignore
    html2canvas(document.getElementById("chart-screenshot-Run"), {logging: false})
      .then(canvas => this.startDownload(this.metric + this.metric2 + this.metric3 + "chart_screenshot.png", canvas.toDataURL("image/png;base64")))
  }

  takeScreenshotModel() {
    // @ts-ignore
    html2canvas(document.getElementById("chart-screenshot-Model"), {logging: false})
      .then(canvas => this.startDownload(this.shownRun + "chart_screenshot.png", canvas.toDataURL("image/png;base64")))
  }

  downloadCSVRuns() {
    let csvString = "runName"
    csvString = csvString + ", " + this.metric1
    csvString = csvString + ", " + this.metric2
    csvString = csvString + ", " + this.metric3
    csvString = csvString + "\n"
    for (const runName in this.totalMetricData[0]) {
      for (const modelName in this.totalMetricData[0][runName]) {
        if (modelName.startsWith("FALCC")) {
          csvString = csvString + runName
          csvString = csvString + ", " + this.totalMetricData[0][runName][modelName][this.metric1]
          csvString = csvString + ", " + this.totalMetricData[0][runName][modelName][this.metric2]
          csvString = csvString + ", " + this.totalMetricData[0][runName][modelName][this.metric3]
          csvString = csvString + "\n"
        }
      }
    }
    const blob = new Blob([csvString], {type: 'text/csv;charset=utf-8;'});
    this.startDownload("AverageResults", URL.createObjectURL(blob));
  }

  downloadCSVSubmodels() {
    if (this.shownRun) {
      let csvString = "model, " + this.metric + "\n"
      for (const modelName in this.totalMetricData[0][this.shownRun]) {
        csvString = csvString + modelName + ", " + this.totalMetricData[0][this.shownRun][modelName][this.metric]
        csvString = csvString + "\n"
      }
      const blob = new Blob([csvString], {type: 'text/csv;charset=utf-8;'});
      this.startDownload("AverageResults", URL.createObjectURL(blob));
    }

  }


  // @ts-ignore
  startDownload(name, href) {
    let htmlElement = document.createElement("a");
    htmlElement.href = href;
    htmlElement.download = name;
    htmlElement.click();
  }
}
