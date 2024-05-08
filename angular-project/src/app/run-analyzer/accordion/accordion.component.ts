import { Component } from '@angular/core';
import {SharedDataService} from "../../shared-data.service";
import {HttpService} from "../../http.service";
import {ToastrService} from "ngx-toastr";
import {Chart} from "chart.js";

@Component({
  selector: 'app-accordion',
  templateUrl: './accordion.component.html',
  styleUrls: ['./accordion.component.scss']
})
export class AccordionComponent {
  runsTrained: boolean = false;
  totalMetricData: any | null = null;
  selectedRun = "run1"
  allRuns = null
  modelNames: any
  recommendations: any
  metrics: string[] = []
  cluster: any = {}
  sensitive: string[] = []
    constructor(private sharedDataService: SharedDataService) {
  }
    ngOnInit() {
    this.sharedDataService.sharedMetricData$.subscribe(data => {
      // store the data
      if (data["runNames"]) {
        this.runsTrained = true;
        this.allRuns = data["runNames"];
      }
      if(data["modelNames"]){
        this.modelNames = data["modelNames"];
      }
      if(data["cluster"]){
        this.cluster = data["cluster"];
      }
      if(data["sensitive"]){
        this.sensitive = data["sensitive"];
      }
      this.totalMetricData = data["results"];
      console.log("metric data = ", this.totalMetricData)
      const firstRun = Object.keys(this.totalMetricData[0])[0];
      const firstModelName = Object.keys(this.totalMetricData[0][firstRun])[0];
      const attributeObject = this.totalMetricData[0][firstRun][firstModelName]

      for (let metric in attributeObject) {
        this.metrics.push(metric)
      }
    });
  }


  showInformationModel(modalName: string) {
  const modelDiv = document.getElementById(modalName)
    if(modelDiv != null){
      modelDiv.style.display = 'block';
    }
  }


  closeInformationModal(modalName: string) {
  const modelDiv = document.getElementById(modalName)
    if(modelDiv != null){
      modelDiv.style.display = 'none';
    }
  }
}
