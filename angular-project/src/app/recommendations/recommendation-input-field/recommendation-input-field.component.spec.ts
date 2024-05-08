import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RecommendationInputFieldComponent } from './recommendation-input-field.component';

describe('RecommendationInputFieldComponent', () => {
  let component: RecommendationInputFieldComponent;
  let fixture: ComponentFixture<RecommendationInputFieldComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [RecommendationInputFieldComponent]
    });
    fixture = TestBed.createComponent(RecommendationInputFieldComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
