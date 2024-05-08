import {Component, Input, OnChanges, OnInit} from '@angular/core';
import {HttpService} from "../../../http.service";
import {ToastrService} from "ngx-toastr";
import { FormsModule } from '@angular/forms';


@Component({
  selector: 'app-cluster-data-accordion-item',
  templateUrl: './cluster-data-accordion-item.component.html',
  styleUrls: ['./cluster-data-accordion-item.component.scss']
})
export class ClusterDataAccordionItemComponent implements OnInit, OnChanges{
  selectedCluster: any;
  @Input() totalMetricData: any | null = null;
  runNames: string[] = []
  @Input() cluster: any = {}
  @Input() sensitive: any = {}
  runName: string = "";
  tableData: any[] = [];
  newMetric: string = "demographic_parity";
  newWeight: number = 0.5;
  usedModelcombination:string[] = []

  ngOnInit() {
    this.runName = this.totalMetricData[1][0]
    this.runNames = Object.keys(this.totalMetricData[1])
    this.runName = this.runNames[0]
  }
  ngOnChanges(){
    this.runName = this.totalMetricData[1][0]
    this.runNames = Object.keys(this.totalMetricData[1])
    this.runName = this.runNames[0]
  }




  constructor(private httpService: HttpService, private toastr: ToastrService) {
  }

  onRunChange(){
    this.selectedCluster = NaN;
  }

  getClusterInformation() {
    this.httpService.getClusterInformation(this.selectedCluster, this.runName).subscribe(
      (response: any) => {
        this.tableData = response;
      }
    )
    this.getBestModelCombination();
  }

  getBestModelCombination(){
    console.log(this.newWeight)
    this.httpService.getModelCombination(this.selectedCluster, this.runName, this.newWeight, this.newMetric).subscribe(
      (response: any) =>{
        this.usedModelcombination = response
        console.log(this.usedModelcombination)
      }
    )
  }

}
