import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RecommendationAccordionItemComponent } from './recommendation-accordion-item.component';

describe('RecommendationAccordionItemComponent', () => {
  let component: RecommendationAccordionItemComponent;
  let fixture: ComponentFixture<RecommendationAccordionItemComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [RecommendationAccordionItemComponent]
    });
    fixture = TestBed.createComponent(RecommendationAccordionItemComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
