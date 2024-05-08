import {Component, Input, OnInit} from '@angular/core';
import {HttpService} from "../../../http.service";


@Component({
  selector: 'app-recommendation-accordion-item',
  templateUrl: './recommendation-accordion-item.component.html',
  styleUrls: ['./recommendation-accordion-item.component.scss']
})
export class RecommendationAccordionItemComponent implements OnInit {
  recommendations: any
  @Input() allRuns: string[] | null = null;
  headers: string[] = []
  shownRun: any;

  constructor(private httpService: HttpService) {
  }

  ngOnInit() {
    if (this.allRuns) {
      this.shownRun = this.allRuns[0]
      this.httpService.getSpecificRecommendations(this.shownRun).subscribe(
        (response: any) => {
          this.recommendations = {}
          this.recommendations = response["recommendations"]
        }
      )
      this.headers = Object.keys(this.recommendations[0])
    }
  }

  getRecommendations(){
    this.httpService.getSpecificRecommendations(this.shownRun).subscribe(
      (response: any) => {
        this.recommendations = {}
        this.recommendations = response["recommendations"]
      }
    )
    this.headers = Object.keys(this.recommendations[0])
  }
}
