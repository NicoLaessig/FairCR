import { Component } from '@angular/core';
import {HttpService} from "../../http.service";
import {SharedDataService} from "../../shared-data.service";
import {ToastrService} from "ngx-toastr";

@Component({
  selector: 'app-recommendation-input-field',
  templateUrl: './recommendation-input-field.component.html',
  styleUrls: ['./recommendation-input-field.component.scss']
})
export class RecommendationInputFieldComponent {
  configuration: any = {
    "datasize": 1200,
    "metric": "demographic_parity",
    "lambda": 0.5,
    "localLambda": 0.5,
    "runTime": 0.5,
    "memory": 0.5,
    "tuning": false
};
  recommendations: any
  toastRef: any;
  constructor(private httpService: HttpService, private toastr: ToastrService) {}
  dataIsLoaded: any;
  datasets: any;
  recommendationReady: any;

  submitData() {
    this.toastRef = this.toastr.info("Calculate recommendations")
    this.httpService.getGeneralRecommendations(this.configuration.metric, this.configuration.tuning,
        this.configuration.lambda, this.configuration.localLambda, this.configuration.datasize, this.configuration.memory, this.configuration.runTime).subscribe(
        (response: any) =>{
                        this.recommendations = {}
                        this.recommendations = response["recommendations"]
                        this.recommendationReady = true
                        this.toastr.clear(this.toastRef.toastId)
            }
    )
  }

    closeInformationModalInput() {
        const modelDiv = document.getElementById("informationModalRecInput")
        if (modelDiv != null) {
            modelDiv.style.display = 'none';
        }
    }

    showInformationModel() {
        const modelDiv = document.getElementById("informationModalRecInput")
        if (modelDiv != null) {
            modelDiv.style.display = 'block';
        }
    }
}
